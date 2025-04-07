import os
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_async_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)

Base = declarative_base()


class TodoList(Base):
    __tablename__ = "todo_lists"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    items = relationship("Item", back_populates="todo_list")


class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    text = Column(String)
    is_done = Column(Boolean, default=False)
    todo_list_id = Column(Integer, ForeignKey('todo_lists.id'))
    todo_list = relationship("TodoList", back_populates="items")


class TodoListCreate(BaseModel):
    name: str


class ItemCreate(BaseModel):
    name: str
    text: str
    todo_list_id: int


app = FastAPI()


async def get_db():
    async with SessionLocal() as session:
        yield session


@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.post("/todo_lists/", response_model=TodoListCreate)
async def create_todo_list(todo_list: TodoListCreate, db: AsyncSession = Depends(get_db)):
    new_todo_list = TodoList(name=todo_list.name)
    db.add(new_todo_list)
    await db.commit()
    await db.refresh(new_todo_list)
    return new_todo_list


@app.get("/todo_lists/{todo_list_id}", response_model=TodoListCreate)
async def read_todo_list(todo_list_id: int, db: AsyncSession = Depends(get_db)):
    todo_list = await db.get(TodoList, todo_list_id)
    if todo_list is None:
        raise HTTPException(status_code=404, detail="Todo list not found")
    return todo_list


@app.patch("/todo_lists/{todo_list_id}", response_model=TodoListCreate)
async def update_todo_list(todo_list_id: int, todo_list: TodoListCreate, db: AsyncSession = Depends(get_db)):
    existing_todo_list = await db.get(TodoList, todo_list_id)
    if existing_todo_list is None:
        raise HTTPException(status_code=404, detail="Todo list not found")

    existing_todo_list.name = todo_list.name
    db.add(existing_todo_list)
    await db.commit()
    await db.refresh(existing_todo_list)
    return existing_todo_list


@app.delete("/todo_lists/{todo_list_id}", response_model=dict)
async def delete_todo_list(todo_list_id: int, db: AsyncSession = Depends(get_db)):
    todo_list = await db.get(TodoList, todo_list_id)
    if todo_list is None:
        raise HTTPException(status_code=404, detail="Todo list not found")

    await db.delete(todo_list)
    await db.commit()
    return {"detail": "Todo list deleted"}


@app.post("/items/", response_model=ItemCreate)
async def create_item(item: ItemCreate, db: AsyncSession = Depends(get_db)):
    new_item = Item(name=item.name, text=item.text, todo_list_id=item.todo_list_id)
    db.add(new_item)
    await db.commit()
    await db.refresh(new_item)
    return new_item

@app.get("/items/{item_id}", response_model=ItemCreate)
async def read_item(item_id: int, db: AsyncSession = Depends(get_db)):
    item = await db.get(Item, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

@app.patch("/items/{item_id}", response_model=ItemCreate)
async def update_item(item_id: int, item: ItemCreate, db: AsyncSession = Depends(get_db)):
    existing_item = await db.get(Item, item_id)
    if existing_item is None:
        raise HTTPException(status_code=404, detail="Item not found")

    existing_item.name = item.name
    existing_item.text = item.text
    existing_item.todo_list_id = item.todo_list_id
    db.add(existing_item)
    await db.commit()
    await db.refresh(existing_item)
    return existing_item

@app.delete("/items/{item_id}", response_model=dict)
async def delete_item(item_id: int, db: AsyncSession = Depends(get_db)):
    item = await db.get(Item, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")

    await db.delete(item)
    await db.commit()
    return {"detail": "Item deleted"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
