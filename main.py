import os
import json
import random
import string
from PIL import Image, ImageDraw, ImageFont


def random_string(length):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


def create_table_image(width, height, rows, cols, border_width, cell_color, border_color, font):
    table_img = Image.new('RGB', (width, height), cell_color)
    draw = ImageDraw.Draw(table_img)

    # Invisible borders: Set border_width to 0.
    if border_width == 0:
        border_color = cell_color

    for i in range(rows + 1):
        y = i * (height // rows)
        draw.line([(0, y), (width, y)], fill=border_color, width=border_width)

    for i in range(cols + 1):
        x = i * (width // cols)
        draw.line([(x, 0), (x, height)], fill=border_color, width=border_width)

    header_color = (200, 200, 255)
    footer_color = (200, 255, 200)
    highlight_color = (255, 255, 200)

    for i in range(rows):
        for j in range(cols):
            cell_text = random_string(random.randint(3, 10))
            text_width, text_height = draw.textsize(cell_text, font=font)
            x = j * (width // cols) + (width // cols - text_width) // 2
            y = i * (height // rows) + (height // rows - text_height) // 2

            if random.random() < 0.1:
                cell_rect = [(j * (width // cols), i * (height // rows)),
                             ((j + 1) * (width // cols), (i + 1) * (height // rows))]
                draw.rectangle(cell_rect, fill=highlight_color)

            draw.text((x, y), cell_text, fill=(0, 0, 0), font=font)

            # Headers and footers: Set the first and last row colors
            if i == 0:
                cell_rect = [(j * (width // cols), i * (height // rows)),
                             ((j + 1) * (width // cols), (i + 1) * (height // rows))]
                draw.rectangle(cell_rect, fill=header_color)
                draw.text((x, y), cell_text, fill=(0, 0, 0), font=font)

            if i == rows - 1:
                cell_rect = [(j * (width // cols), i * (height // rows)),
                             ((j + 1) * (width // cols), (i + 1) * (height // rows))]
                draw.rectangle(cell_rect, fill=footer_color)
                draw.text((x, y), cell_text, fill=(0, 0, 0), font=font)

    return table_img


def create_text_image(width, height, font, num_lines):
    text_img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(text_img)

    for i in range(num_lines):
        line_text = random_string(random.randint(20, 50))
        y = i * (height // num_lines)
        draw.text((0, y), line_text, fill=(0, 0, 0), font=font)

    return text_img


def generate_synthetic_tables(output_folder, num_images):
    os.makedirs(output_folder, exist_ok=True)
    # font = ImageFont.truetype('arial.ttf', 14)
    font = ImageFont.load_default()

    annotations = []

    for i in range(num_images):
        table_width = random.randint(200, 600)
        table_height = random.randint(200, 600)
        rows = random.randint(2, 10)
        cols = random.randint(2, 10)
        border_width = random.randint(1, 5)
        cell_color = (random.randint(200, 255), random.randint(
            200, 255), random.randint(200, 255))
        border_color = (random.randint(0, 150), random.randint(
            0, 150), random.randint(0, 150))

        table_img = create_table_image(
            table_width, table_height, rows, cols, border_width, cell_color, border_color, font)

        text_width = table_width
        text_height = random.randint(50, 200)
        num_lines = random.randint(2, 6)

        # Create text image (before and after table)
        text_img_before = create_text_image(
            text_width, text_height, font, num_lines)
        text_img_after = create_text_image(
            text_width, text_height, font, num_lines)

        # Combine text and table images
        final_img = Image.new(
            'RGB', (text_width, text_height * 2 + table_height), (255, 255, 255))
        final_img.paste(text_img_before, (0, 0))
        final_img.paste(table_img, (0, text_height))
        final_img.paste(text_img_after, (0, text_height + table_height))

        output_path = os.path.join(output_folder, f'table_image_{i+1}.png')
        final_img.save(output_path)

        annotation = {
            "filename": f'table_image_{i+1}.png',
            "annotations": [
                {
                    "label": "table",
                    "coordinates": {"x": 0, "y": text_height, "width": table_width, "height": table_height},
                    "cells": [
                        {
                            "label": "cell",
                            "coordinates": {"x": col * (table_width // cols), "y": text_height + row * (table_height // rows), "width": table_width // cols, "height": table_height // rows}
                        }
                        for row in range(rows) for col in range(cols)
                    ]
                }
            ]
        }
        annotations.append(annotation)

    with open(os.path.join(output_folder, 'annotations.json'), 'w') as outfile:
        json.dump(annotations, outfile, indent=4)


if __name__ == "__main__":
    output_folder = 'synthetic_tables'
    num_images = 5
    generate_synthetic_tables(output_folder, num_images)
