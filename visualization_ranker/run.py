"""Run the ranker app. Uses port 8080 by default; change if blocked."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=True)
