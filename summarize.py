from dotenv import load_dotenv
from langchain.document_loaders import Docx2txtLoader
from langchain.llms import YandexGPT
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

import textwrap
import dotenv
from dotenv import load_dotenv

load_dotenv()
config = dotenv.dotenv_values()

import streamlit as st


def process_docx(docx_file):
    # Добавьте ваш код обработки docx здесь
    text = ""
    # Docx2txtLoader загружает документ
    loader = Docx2txtLoader(docx_file)

    # Загружает документы и разделяет на части
    text = loader.load_and_split()

    return text


def process_pdf(pdf_file):
    text = ""
    # PYPDFLoader загружает список объектов PDF Document
    loader = PyPDFLoader(pdf_file)
    pages = loader.load()

    for page in pages:
        text += page.page_content
    text = text.replace('\t', ' ')

    # Разделяет длинный документ на меньшие части, которые помещаются в контекстное окно модели LLM
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=50
    )
    # create_documents() создает документы из списка текстов
    texts = text_splitter.create_documents([text])

    print(len(text))

    return texts


def main():
    st.title("Генератор резюме")

    uploaded_file = st.file_uploader("Выберите резюме", type=["docx", "pdf"])
    text = ""
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1]

        st.write("Детали файла:")
        st.write(f"Имя файла: {uploaded_file.name}")
        st.write(f"Тип файла: {file_extension}")

        if file_extension == "docx":
            text = process_docx("data/"+uploaded_file.name)
        elif file_extension == "pdf":
            text = process_pdf("data/"+uploaded_file.name)
        else:
            st.error("Неподдерживаемый формат файла. Пожалуйста, загрузите файл .docx или .pdf.")
            return

        llm = YandexGPT(
            model_name="yandexgpt",
            temperature=0.0,
        )

        prompt_template = """Вам предоставлено резюме для анализа. 
        Напишите подробный анализ следующего: 
        {text}
        Детали:"""
        prompt = PromptTemplate.from_template(prompt_template)

        refine_template = (
            "Ваша задача - создать итоговый результат\n"
            "Мы предоставили существующий анализ: {existing_answer}\n"
            "Мы хотим уточненную версию существующего анализа на основе начальных деталей ниже\n"
            "------------\n"
            "{text}\n"
            "------------\n"
            "Учитывая новый контекст, уточните оригинальное резюме следующим образом:"
            "Имя: \n"
            "Email: \n"
            "Ключевые навыки: \n"
            "Последняя компания: \n"
            "Опыт работы: \n"
        )
        refine_prompt = PromptTemplate.from_template(refine_template)
        chain = load_summarize_chain(
            llm=llm,
            chain_type="refine",
            question_prompt=prompt,
            refine_prompt=refine_prompt,
            return_intermediate_steps=True,
            input_key="input_documents",
            output_key="output_text",
        )
        result = chain({"input_documents": text}, return_only_outputs=True)

        st.write("Анализ резюме:")
        st.text_area("Текст", result['output_text'], height=400)


if __name__ == "__main__":
    main()

