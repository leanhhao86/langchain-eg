from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from rag_conversation import chain as rag_conversation_chain
from dotenv import load_dotenv


from pydantic import BaseModel, Field
from typing import List, Tuple, Optional

app = FastAPI()

# User input
class ChatHistory(BaseModel):
    chat_history: Optional[List[Tuple[str, str]]] = Field(None, extra={"widget": {"type": "chat"}})
    question: str

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

add_routes(app, rag_conversation_chain, path="/rag-conversation")

# @app.post("/query/")
# async def make_query(chat: ChatHistory):
#     return rag_conversation_chain.invoke({
#         "question": chat.question,
#         "chat_history": chat.chat_history
#         })

if __name__ == "__main__":
    import uvicorn
    load_dotenv()
    uvicorn.run(app, host="0.0.0.0", port=8000)
