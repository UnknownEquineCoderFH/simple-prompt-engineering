from __future__ import annotations

import dotenv
import openai

import typer

from dataclasses import dataclass
from typing import TypedDict


class Message(TypedDict, total=False):
    role: str
    content: str


class Choice(TypedDict, total=False):
    index: int
    text: str
    message: Message


class ChatCompletion(TypedDict, total=False):
    id: str
    object: str
    created: int
    model: str
    choices: list[Choice]
    messages: list[dict[str, str]]


class Image(TypedDict, total=False):
    id: str
    object: str
    created: int
    model: str
    data: list[ImageData]


class ImageData(TypedDict, total=False):
    url: str


def enrich_prompt(prompt: str, length: int, temperature: float) -> str:
    response: ChatCompletion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=length,
        temperature=temperature,
        top_p=1,
        frequency_penalty=0,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )  # type: ignore

    return (
        response.get("choices", [{}])[0]
        .get("message", {"content": "No content"})
        .get("content", "No content")
    )


def generate_image(prompt: str, size: int) -> str:
    image: Image = openai.Image.create(prompt=prompt, n=1, size=f"{size}x{size}")  # type: ignore

    return image.get("data", [{}])[0].get("url", "No image")


def main(
    prompt: str,
    length: int = 200,
    temperature: float = 0.25,
    size: int = 512,
) -> int:
    vars = dotenv.dotenv_values()

    key = vars.get("OPENAI_KEY", "OpenAI key not found")

    openai.api_key = key

    extender = vars.get("EXTENDER", "No extender found")

    full_prompt = (
        f'{prompt} {enrich_prompt(f"{extender} {prompt}", length, temperature)}'
    )

    print(f"Generating: {full_prompt!r}")

    print(f"URL: {generate_image(full_prompt, size)}")

    return 0


if __name__ == "__main__":
    typer.run(main)
