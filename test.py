import httpx
import asyncio

async def test_api(image_path: str):
    url = "http://127.0.0.1:8000/process_image"
    payload = {"image_path": image_path}

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
            print(f"Response status: {response.status_code}")
            print(f"Response data: {response.json()}")
        except httpx.RequestError as exc:
            print(f"An error occurred while requesting {exc.request.url!r}.")
            print(f"Exception: {exc}")
        except httpx.HTTPStatusError as exc:
            print(f"Non-successful status code: {exc.response.status_code} for URL: {exc.request.url!r}")
            print(f"Response data: {exc.response.json()}")

if __name__ == "__main__":
    # Replace 'your_image_path.jpg' with the actual image path on your system
    image_path = '1.jpg'
    asyncio.run(test_api(image_path))
