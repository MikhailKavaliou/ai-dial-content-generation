import base64
from pathlib import Path

from task._utils.constants import API_KEY, DIAL_CHAT_COMPLETIONS_ENDPOINT
from task._utils.model_client import DialModelClient
from task._models.role import Role
from task.image_to_text.openai.message import ContentedMessage, TxtContent, ImgContent, ImgUrl


def start() -> None:
    project_root = Path(__file__).parent.parent.parent.parent
    image_path = project_root / "dialx-banner.png"

    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    # 1. Create DialModelClient
    client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name="gpt-4o",
        api_key=API_KEY
    )

    # 2. Call client to analyse image:
    #    - try with base64 encoded format
    base64_data_url = f"data:image/png;base64,{base64_image}"
    base64_msg = ContentedMessage(
        role=Role.USER,
        content=[
            TxtContent(text="What is in this image?"),
            ImgContent(image_url=ImgUrl(url=base64_data_url))
        ]
    )
    base64_response = client.get_completion([base64_msg])
    print("Base64 image response:", base64_response)

    #    - try with URL: https://a-z-animals.com/media/2019/11/Elephant-male-1024x535.jpg
    image_url = "https://a-z-animals.com/media/2019/11/Elephant-male-1024x535.jpg"
    url_msg = ContentedMessage(
        role=Role.USER,
        content=[
            TxtContent(text="What is in this image?"),
            ImgContent(image_url=ImgUrl(url=image_url))
        ]
    )
    url_response = client.get_completion([url_msg])
    print("Image URL response:", url_response)


start()
