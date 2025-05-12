import os
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime, func
from sqlalchemy.orm import relationship
from sqlalchemy.sql import select
from dotenv import load_dotenv
from pydantic import BaseModel, Field, computed_field

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_async_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)

Base = declarative_base()


class TodoList(Base):
    __tablename__ = "todo_lists"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    deleted_at = Column(DateTime, nullable=True)
    total_items = Column(Integer, default=0)
    completed_items = Column(Integer, default=0)
    items = relationship("Item", back_populates="todo_list")


class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    text = Column(String)
    is_done = Column(Boolean, default=False)
    deleted_at = Column(DateTime, nullable=True)
    todo_list_id = Column(Integer, ForeignKey('todo_lists.id'))
    todo_list = relationship("TodoList", back_populates="items")


class TodoListBase(BaseModel):
    name: str


class TodoListCreate(TodoListBase):
    pass


class TodoListResponse(TodoListBase):
    id: int
    total_items: int
    completed_items: int
    deleted_at: Optional[datetime] = None

    @computed_field
    @property
    def progress(self) -> float:
        if self.total_items > 0:
            return (self.completed_items / self.total_items) * 100
        return 0.0

    class Config:
        from_attributes = True


class ItemBase(BaseModel):
    name: str
    text: str
    todo_list_id: int


class ItemCreate(ItemBase):
    pass


class ItemResponse(ItemBase):
    id: int
    is_done: bool
    deleted_at: Optional[datetime] = None

    class Config:
        from_attributes = True


app = FastAPI()


async def get_db():
    async with SessionLocal() as session:
        yield session


@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.post("/todo_lists/", response_model=TodoListResponse)
async def create_todo_list(todo_list: TodoListCreate, db: AsyncSession = Depends(get_db)):
    new_todo_list = TodoList(name=todo_list.name)
    db.add(new_todo_list)
    await db.commit()
    await db.refresh(new_todo_list)
    return new_todo_list


@app.get("/todo_lists/{todo_list_id}", response_model=TodoListResponse)
async def read_todo_list(todo_list_id: int, db: AsyncSession = Depends(get_db)):
    stmt = select(TodoList).where(TodoList.id == todo_list_id, TodoList.deleted_at == None)
    result = await db.execute(stmt)
    todo_list = result.scalar_one_or_none()
    if todo_list is None:
        raise HTTPException(status_code=404, detail="Todo list not found")

    stmt_items = select(
        func.count(Item.id),
        func.count().filter(Item.is_done == True)
    ).where(
        Item.todo_list_id == todo_list_id,
        Item.deleted_at == None
    )
    result_items = await db.execute(stmt_items)
    total, completed = result_items.first()

    todo_list.total_items = total
    todo_list.completed_items = completed

    return todo_list


@app.patch("/todo_lists/{todo_list_id}", response_model=TodoListResponse)
async def update_todo_list(todo_list_id: int, todo_list: TodoListCreate, db: AsyncSession = Depends(get_db)):
    stmt = select(TodoList).where(TodoList.id == todo_list_id, TodoList.deleted_at == None)
    result = await db.execute(stmt)
    existing_todo_list = result.scalar_one_or_none()
    if existing_todo_list is None:
        raise HTTPException(status_code=404, detail="Todo list not found")

    existing_todo_list.name = todo_list.name
    db.add(existing_todo_list)
    await db.commit()
    await db.refresh(existing_todo_list)
    return existing_todo_list


@app.delete("/todo_lists/{todo_list_id}", response_model=dict)
async def delete_todo_list(todo_list_id: int, db: AsyncSession = Depends(get_db)):
    stmt = select(TodoList).where(TodoList.id == todo_list_id, TodoList.deleted_at == None)
    result = await db.execute(stmt)
    todo_list = result.scalar_one_or_none()
    if todo_list is None:
        raise HTTPException(status_code=404, detail="Todo list not found")

    todo_list.deleted_at = datetime.utcnow()
    db.add(todo_list)
    await db.commit()
    return {"detail": "Todo list soft deleted"}


@app.post("/items/", response_model=ItemResponse)
async def create_item(item: ItemCreate, db: AsyncSession = Depends(get_db)):
    stmt = select(TodoList).where(TodoList.id == item.todo_list_id, TodoList.deleted_at == None)
    result = await db.execute(stmt)
    todo_list = result.scalar_one_or_none()
    if todo_list is None:
        raise HTTPException(status_code=404, detail="Todo list not found")

    new_item = Item(
        name=item.name,
        text=item.text,
        todo_list_id=item.todo_list_id
    )
    db.add(new_item)

    todo_list.total_items += 1
    db.add(todo_list)

    await db.commit()
    await db.refresh(new_item)
    return new_item


@app.get("/items/{item_id}", response_model=ItemResponse)
async def read_item(item_id: int, db: AsyncSession = Depends(get_db)):
    stmt = select(Item).where(Item.id == item_id, Item.deleted_at == None)
    result = await db.execute(stmt)
    item = result.scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


@app.patch("/items/{item_id}", response_model=ItemResponse)
async def update_item(item_id: int, item: ItemCreate, db: AsyncSession = Depends(get_db)):
    stmt = select(Item).where(Item.id == item_id, Item.deleted_at == None)
    result = await db.execute(stmt)
    existing_item = result.scalar_one_or_none()
    if existing_item is None:
        raise HTTPException(status_code=404, detail="Item not found")

    stmt_todo = select(TodoList).where(TodoList.id == item.todo_list_id, TodoList.deleted_at == None)
    result_todo = await db.execute(stmt_todo)
    todo_list = result_todo.scalar_one_or_none()
    if todo_list is None:
        raise HTTPException(status_code=404, detail="Todo list not found")

    if existing_item.todo_list_id != item.todo_list_id:
        old_todo_stmt = select(TodoList).where(TodoList.id == existing_item.todo_list_id)
        old_todo_result = await db.execute(old_todo_stmt)
        old_todo_list = old_todo_result.scalar_one_or_none()
        if old_todo_list:
            old_todo_list.total_items -= 1
            if existing_item.is_done:
                old_todo_list.completed_items -= 1
            db.add(old_todo_list)

        todo_list.total_items += 1
        if existing_item.is_done:
            todo_list.completed_items += 1
        db.add(todo_list)

    existing_item.name = item.name
    existing_item.text = item.text
    existing_item.todo_list_id = item.todo_list_id
    db.add(existing_item)
    await db.commit()
    await db.refresh(existing_item)
    return existing_item


@app.patch("/items/{item_id}/toggle", response_model=ItemResponse)
async def toggle_item(item_id: int, db: AsyncSession = Depends(get_db)):
    stmt = select(Item).where(Item.id == item_id, Item.deleted_at == None)
    result = await db.execute(stmt)
    item = result.scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")

    stmt_todo = select(TodoList).where(TodoList.id == item.todo_list_id, TodoList.deleted_at == None)
    result_todo = await db.execute(stmt_todo)
    todo_list = result_todo.scalar_one_or_none()
    if todo_list is None:
        raise HTTPException(status_code=404, detail="Todo list not found")

    if item.is_done:
        todo_list.completed_items -= 1
    else:
        todo_list.completed_items += 1

    item.is_done = not item.is_done
    db.add(item)
    db.add(todo_list)
    await db.commit()
    await db.refresh(item)
    return item


@app.delete("/items/{item_id}", response_model=dict)
async def delete_item(item_id: int, db: AsyncSession = Depends(get_db)):
    stmt = select(Item).where(Item.id == item_id, Item.deleted_at == None)
    result = await db.execute(stmt)
    item = result.scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")

    stmt_todo = select(TodoList).where(TodoList.id == item.todo_list_id, TodoList.deleted_at == None)
    result_todo = await db.execute(stmt_todo)
    todo_list = result_todo.scalar_one_or_none()
    if todo_list is None:
        raise HTTPException(status_code=404, detail="Todo list not found")

    todo_list.total_items -= 1
    if item.is_done:
        todo_list.completed_items -= 1

    item.deleted_at = datetime.utcnow()
    db.add(item)
    db.add(todo_list)
    await db.commit()
    return {"detail": "Item soft deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("lab3:app", host="0.0.0.0", port=8000, reload=True)
