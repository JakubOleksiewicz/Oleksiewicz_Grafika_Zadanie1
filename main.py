
import urllib.request
import cv2
import matplotlib.pyplot as plt
import numpy as np

IMAGE_URL = (
    "https://thumb.wikimedia.org/wikipedia/commons/thumb/3/3c/"
    "2021-09-13_16_40_16_A_lemon_and_lime_in_the_Franklin_Farm_section_of_Oak_Hill%2C_Fairfax_County%2C_Virginia.jpg/"
    "960px-2021-09-13_16_40_16_A_lemon_and_lime_in_the_Franklin_Farm_section_of_Oak_Hill%2C_Fairfax_County%2C_Virginia.jpg"
)
USER_AGENT = "Mozilla/5.0"


def load_image(url):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT}
    )

    with urllib.request.urlopen(request, timeout=20) as response:
        image_data = response.read()

    image_array = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Nie udało się zdekodować pobranego obrazu")

    return image


def display_image(image, grayscale=False):
    if not grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(frameon=False)
    plt.imshow(
        image,
        cmap="gray" if grayscale else None,
        vmin=0 if grayscale else None,
        vmax=255 if grayscale else None
    )
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()


def process_image(image):
    height, width = image.shape[:2]

    new_width = width // 2
    new_height = height // 2

    processed_image = cv2.resize(image, (new_width, new_height))
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)

    return processed_image


def display_matrix(image):
    np.set_printoptions(threshold=np.inf, linewidth=120)

    print("\nMacierz obrazu wynikowego:")
    print(image)


def main():
    try:
        image = load_image(IMAGE_URL)

        display_image(image)

        processed_image = process_image(image)

        display_image(processed_image)
        display_matrix(processed_image)

    except Exception as error:
        print(f"Wystąpił błąd: {error}")


if __name__ == "__main__":
    main()
