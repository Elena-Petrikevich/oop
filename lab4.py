import os
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime, func
from sqlalchemy.orm import relationship
from sqlalchemy.sql import select, update
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


class TodoListAggregate:
    def __init__(self, id: int, name: str, items: List['ItemAggregate'] = None):
        self.id = id
        self.name = name
        self.items = items or []
        self.deleted_at = None

    def add_item(self, item: 'ItemAggregate'):
        self.items.append(item)

    def update_item(self, item_id: int, name: str, text: str, is_done: bool = None):
        for item in self.items:
            if item.id == item_id:
                item.name = name
                item.text = text
                if is_done is not None:
                    item.is_done = is_done
                return
        raise ValueError("Item not found")

    def delete_item(self, item_id: int):
        for item in self.items:
            if item.id == item_id:
                item.deleted_at = datetime.utcnow()
                return
        raise ValueError("Item not found")

    def toggle_item(self, item_id: int):
        for item in self.items:
            if item.id == item_id:
                item.is_done = not item.is_done
                return
        raise ValueError("Item not found")

    def mark_as_deleted(self):
        self.deleted_at = datetime.utcnow()
        for item in self.items:
            item.deleted_at = self.deleted_at


class ItemAggregate:
    def __init__(self, id: int, name: str, text: str, todo_list_id: int, is_done: bool = False):
        self.id = id
        self.name = name
        self.text = text
        self.is_done = is_done
        self.todo_list_id = todo_list_id
        self.deleted_at = None


class TodoListRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get(self, todo_list_id: int) -> Optional[TodoListAggregate]:
        stmt = select(TodoList).where(TodoList.id == todo_list_id, TodoList.deleted_at == None)
        result = await self.db.execute(stmt)
        todo_list = result.scalar_one_or_none()
        if not todo_list:
            return None

        stmt_items = select(Item).where(Item.todo_list_id == todo_list_id, Item.deleted_at == None)
        result_items = await self.db.execute(stmt_items)
        items = result_items.scalars().all()

        aggregate = TodoListAggregate(
            id=todo_list.id,
            name=todo_list.name,
            items=[ItemAggregate(
                id=item.id,
                name=item.name,
                text=item.text,
                todo_list_id=item.todo_list_id,
                is_done=item.is_done
            ) for item in items]
        )
        aggregate.deleted_at = todo_list.deleted_at
        return aggregate

    async def create(self, name: str) -> TodoListAggregate:
        todo_list = TodoList(name=name)
        self.db.add(todo_list)
        await self.db.commit()
        await self.db.refresh(todo_list)
        return TodoListAggregate(id=todo_list.id, name=todo_list.name)

    async def update(self, todo_list: TodoListAggregate):
        stmt = (
            update(TodoList)
            .where(TodoList.id == todo_list.id)
            .values(name=todo_list.name, deleted_at=todo_list.deleted_at)
        )
        await self.db.execute(stmt)

        for item in todo_list.items:
            if item.id:
                stmt = (
                    update(Item)
                    .where(Item.id == item.id)
                    .values(
                        name=item.name,
                        text=item.text,
                        is_done=item.is_done,
                        deleted_at=item.deleted_at,
                        todo_list_id=item.todo_list_id
                    )
                )
            else:
                new_item = Item(
                    name=item.name,
                    text=item.text,
                    is_done=item.is_done,
                    todo_list_id=item.todo_list_id,
                    deleted_at=item.deleted_at
                )
                self.db.add(new_item)

        await self.db.commit()

    async def delete(self, todo_list_id: int):
        stmt = (
            update(TodoList)
            .where(TodoList.id == todo_list_id)
            .values(deleted_at=datetime.utcnow())
        )
        await self.db.execute(stmt)

        stmt_items = (
            update(Item)
            .where(Item.todo_list_id == todo_list_id)
            .values(deleted_at=datetime.utcnow())
        )
        await self.db.execute(stmt_items)
        await self.db.commit()


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


async def get_todo_list_repository(db: AsyncSession = Depends(get_db)):
    return TodoListRepository(db)


@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.post("/todo_lists/", response_model=TodoListResponse)
async def create_todo_list(
        todo_list: TodoListCreate,
        repo: TodoListRepository = Depends(get_todo_list_repository)
):
    aggregate = await repo.create(todo_list.name)

    async with SessionLocal() as db:
        stmt_items = select(
            func.count(Item.id),
            func.count().filter(Item.is_done == True)
        ).where(
            Item.todo_list_id == aggregate.id,
            Item.deleted_at == None
        )
        result_items = await db.execute(stmt_items)
        total, completed = result_items.first()

        return {
            "id": aggregate.id,
            "name": aggregate.name,
            "total_items": total,
            "completed_items": completed,
            "deleted_at": None
        }


@app.get("/todo_lists/{todo_list_id}", response_model=TodoListResponse)
async def read_todo_list(
        todo_list_id: int,
        repo: TodoListRepository = Depends(get_todo_list_repository)
):
    aggregate = await repo.get(todo_list_id)
    if not aggregate:
        raise HTTPException(status_code=404, detail="Todo list not found")

    total = len(aggregate.items)
    completed = sum(1 for item in aggregate.items if item.is_done)

    return {
        "id": aggregate.id,
        "name": aggregate.name,
        "total_items": total,
        "completed_items": completed,
        "deleted_at": aggregate.deleted_at
    }


@app.patch("/todo_lists/{todo_list_id}", response_model=TodoListResponse)
async def update_todo_list(
        todo_list_id: int,
        todo_list: TodoListCreate,
        repo: TodoListRepository = Depends(get_todo_list_repository)
):
    aggregate = await repo.get(todo_list_id)
    if not aggregate:
        raise HTTPException(status_code=404, detail="Todo list not found")

    aggregate.name = todo_list.name
    await repo.update(aggregate)

    total = len(aggregate.items)
    completed = sum(1 for item in aggregate.items if item.is_done)

    return {
        "id": aggregate.id,
        "name": aggregate.name,
        "total_items": total,
        "completed_items": completed,
        "deleted_at": aggregate.deleted_at
    }


@app.delete("/todo_lists/{todo_list_id}", response_model=dict)
async def delete_todo_list(
        todo_list_id: int,
        repo: TodoListRepository = Depends(get_todo_list_repository)
):
    aggregate = await repo.get(todo_list_id)
    if not aggregate:
        raise HTTPException(status_code=404, detail="Todo list not found")

    await repo.delete(todo_list_id)
    return {"detail": "Todo list soft deleted"}


@app.post("/items/", response_model=ItemResponse)
async def create_item(
        item: ItemCreate,
        repo: TodoListRepository = Depends(get_todo_list_repository)
):
    aggregate = await repo.get(item.todo_list_id)
    if not aggregate:
        raise HTTPException(status_code=404, detail="Todo list not found")

    new_item = ItemAggregate(
        id=None,
        name=item.name,
        text=item.text,
        todo_list_id=item.todo_list_id
    )
    aggregate.add_item(new_item)
    await repo.update(aggregate)

    async with SessionLocal() as db:
        stmt = select(Item).where(
            Item.todo_list_id == item.todo_list_id,
            Item.name == item.name,
            Item.text == item.text,
            Item.deleted_at == None
        ).order_by(Item.id.desc())
        result = await db.execute(stmt)
        db_item = result.scalars().first()

    return {
        "id": db_item.id,
        "name": db_item.name,
        "text": db_item.text,
        "todo_list_id": db_item.todo_list_id,
        "is_done": db_item.is_done,
        "deleted_at": db_item.deleted_at
    }


@app.get("/items/{item_id}", response_model=ItemResponse)
async def read_item(item_id: int, db: AsyncSession = Depends(get_db)):
    stmt = select(Item).where(Item.id == item_id, Item.deleted_at == None)
    result = await db.execute(stmt)
    item = result.scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


@app.patch("/items/{item_id}", response_model=ItemResponse)
async def update_item(
        item_id: int,
        item: ItemCreate,
        repo: TodoListRepository = Depends(get_todo_list_repository)
):
    async with SessionLocal() as db:
        stmt = select(Item).where(Item.id == item_id, Item.deleted_at == None)
        result = await db.execute(stmt)
        db_item = result.scalar_one_or_none()
        if not db_item:
            raise HTTPException(status_code=404, detail="Item not found")

    aggregate = await repo.get(db_item.todo_list_id)
    if not aggregate:
        raise HTTPException(status_code=404, detail="Todo list not found")

    try:
        aggregate.update_item(item_id, item.name, item.text)
        await repo.update(aggregate)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "id": item_id,
        "name": item.name,
        "text": item.text,
        "todo_list_id": item.todo_list_id,
        "is_done": db_item.is_done,
        "deleted_at": None
    }


@app.patch("/items/{item_id}/toggle", response_model=ItemResponse)
async def toggle_item(
        item_id: int,
        repo: TodoListRepository = Depends(get_todo_list_repository)
):
    async with SessionLocal() as db:
        stmt = select(Item).where(Item.id == item_id, Item.deleted_at == None)
        result = await db.execute(stmt)
        db_item = result.scalar_one_or_none()
        if not db_item:
            raise HTTPException(status_code=404, detail="Item not found")

    aggregate = await repo.get(db_item.todo_list_id)
    if not aggregate:
        raise HTTPException(status_code=404, detail="Todo list not found")

    try:
        aggregate.toggle_item(item_id)
        await repo.update(aggregate)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "id": item_id,
        "name": db_item.name,
        "text": db_item.text,
        "todo_list_id": db_item.todo_list_id,
        "is_done": not db_item.is_done,
        "deleted_at": None
    }


@app.delete("/items/{item_id}", response_model=dict)
async def delete_item(
        item_id: int,
        repo: TodoListRepository = Depends(get_todo_list_repository)
):
    async with SessionLocal() as db:
        stmt = select(Item).where(Item.id == item_id, Item.deleted_at == None)
        result = await db.execute(stmt)
        db_item = result.scalar_one_or_none()
        if not db_item:
            raise HTTPException(status_code=404, detail="Item not found")

    aggregate = await repo.get(db_item.todo_list_id)
    if not aggregate:
        raise HTTPException(status_code=404, detail="Todo list not found")

    try:
        aggregate.delete_item(item_id)
        await repo.update(aggregate)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {"detail": "Item soft deleted"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("lab4:app", host="0.0.0.0", port=8000, reload=True)