import asyncio
from datetime import datetime

from task._models.custom_content import Attachment
from task._utils.constants import API_KEY, DIAL_URL, DIAL_CHAT_COMPLETIONS_ENDPOINT
from task._utils.bucket_client import DialBucketClient
from task._utils.model_client import DialModelClient
from task._models.message import Message
from task._models.role import Role

class Size:
    """
    The size of the generated image.
    """
    square: str = '1024x1024'
    height_rectangle: str = '1024x1792'
    width_rectangle: str = '1792x1024'


class Style:
    """
    The style of the generated image. Must be one of vivid or natural.
     - Vivid causes the model to lean towards generating hyper-real and dramatic images.
     - Natural causes the model to produce more natural, less hyper-real looking images.
    """
    natural: str = "natural"
    vivid: str = "vivid"


class Quality:
    """
    The quality of the image that will be generated.
     - ‘hd’ creates images with finer details and greater consistency across the image.
    """
    standard: str = "standard"
    hd: str = "hd"

async def _save_images(attachments: list[Attachment]):
    # 1. Create DIAL bucket client
    async with DialBucketClient(api_key=API_KEY, base_url=DIAL_URL) as client:
        # 2. Iterate through Images from attachments, download them and then save here
        for attachment in attachments:
            if attachment.url:
                # Download the image
                image_data = await client.get_file(attachment.url)

                # Generate a filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Determine file extension from mime type
                file_extension = ''
                if attachment.type:
                    if 'png' in attachment.type.lower():
                        file_extension = '.png'
                    elif 'jpeg' in attachment.type.lower() or 'jpg' in attachment.type.lower():
                        file_extension = '.jpg'
                    elif 'gif' in attachment.type.lower():
                        file_extension = '.gif'
                    elif 'webp' in attachment.type.lower():
                        file_extension = '.webp'
                    else:
                        # Default to png if unknown
                        file_extension = '.png'
                else:
                    file_extension = '.png'

                filename = f"generated_image_{timestamp}_{attachment.title or 'image'}{file_extension}"

                # Save the image locally
                with open(filename, 'wb') as f:
                    f.write(image_data)

                # 3. Print confirmation that image has been saved locally
                print(f"Image saved locally: {filename}")


def start() -> None:
    # 1. Create DialModelClient
    client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name="dall-e-3",
        api_key=API_KEY
    )

    # 2. Generate image for "Sunny day on Bali"
    prompt = "Sunny day on Bali"
    message = Message(
        role=Role.USER,
        content=prompt
    )

    # 4. Try to configure the picture for output via `custom_fields` parameter.
    custom_fields = {
        "size": Size.square,
        "quality": Quality.hd,
        "style": Style.vivid
    }

    print(f"Generating image with prompt: '{prompt}'")
    print(f"Custom fields: {custom_fields}")

    response = client.get_completion(
        messages=[message],
        custom_fields=custom_fields
    )

    print(f"Response: {response}")

    # 3. Get attachments from response and save generated message (use method `_save_images`)
    if response.custom_content and response.custom_content.attachments:
        print(f"Found {len(response.custom_content.attachments)} attachment(s)")
        asyncio.run(_save_images(response.custom_content.attachments))
    else:
        print("No attachments found in response")

    # 5. Test it with the 'imagegeneration@005' (Google image generation model)
    # Note: This may fail if the model is not available in your DIAL deployment
    print("\n--- Testing with Google image generation model ---")
    try:
        client_google = DialModelClient(
            endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
            deployment_name="imagegeneration@005",
            api_key=API_KEY
        )

        response_google = client_google.get_completion(
            messages=[message],
            custom_fields=custom_fields
        )

        print(f"Google Response: {response_google}")

        if response_google.custom_content and response_google.custom_content.attachments:
            print(f"Found {len(response_google.custom_content.attachments)} attachment(s)")
            asyncio.run(_save_images(response_google.custom_content.attachments))
        else:
            print("No attachments found in Google response")
    except Exception as e:
        print(f"Failed to generate image with Google model: {e}")
        print("This is expected if the model is not available in your DIAL deployment")

start()
