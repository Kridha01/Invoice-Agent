import os
from dotenv import load_dotenv
load_dotenv()
import json
import pandas as pd
from typing import List,Dict,DefaultDict

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


class FieldExtractor:
    def __init__(self, pdf_path: str, excel_path: str, api_key: str):
        self.pdf_path = pdf_path
        self.excel_path = excel_path
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        os.environ["GOOGLE_API_KEY"] = api_key
        self.loader = PyPDFLoader(pdf_path)
        self.docs = self.loader.load()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=150,
            chunk_overlap=30,
            add_start_index=True
        )
        self.all_splits = self.text_splitter.split_documents(self.docs)
        # print("splits")
        # print(self.all_splits)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = FAISS.from_documents(self.all_splits, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        self.fields_df = pd.read_excel(excel_path)

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
        )

        system_message = (
            "You are an assistant that extracts structured information from text. "
            "Return strictly in csv format like first row field names separated by '|' and next row values separated by '|'. if the column numbers are 8 then delimitor of '|' must be 7 only do not add a trailing delimitor if no value present."
            "Example:\npolicy|address|state|pin|gst number|amount|net profit\n1233|asb damoadr nagar|UP|208027|234N5|98.00|12.00"
        )
        human_prompt_template = "{fields_text}"

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", human_prompt_template)
        ])

    def retrieve_context(self, field_name: str, field_desc: str) -> str:
        query = f"Provide context for the field: {field_name} ({field_desc})"
        retrieved_docs = self.vectorstore.similarity_search(query, k=6)
        context_text = "\n".join([doc.page_content for doc in retrieved_docs])
        return context_text

    def prepare_fields_text(self, feedback: Dict[str, str] | None = None) -> str:
        all_fields_text = ""
        for _, row in self.fields_df.iterrows():
            field_name = row["Field Name"]
            field_desc = row["Description"]
            context_text = self.retrieve_context(field_name, field_desc)

            field_feedback = ""
            if feedback and field_name in feedback:
                field_feedback = f"\nFeedback from previous evaluation: {feedback[field_name]}"

            all_fields_text += (
                f"Field: {field_name}\n"
                f"Description: {field_desc}"
                f"{field_feedback}\n"
                f"Context:\n{context_text}\n\n"
            )
        # print(all_fields_text)
        return all_fields_text

    def extract_fields(self, feedback: Dict[str, str] | None = None) -> pd.DataFrame:
        fields_text = self.prepare_fields_text(feedback)
        chain = self.prompt | self.llm
        response = chain.invoke({"fields_text": fields_text})

        result = response.content.strip()
        print(result)

        lines = [line for line in result.split("\n") if line.strip()]

        headers = [h.strip() for h in lines[0].split("|")]
        num_cols = len(headers)

        rows = []
        for line in lines[1:]:
            cols = [c.strip() for c in line.split("|")]

            if len(cols) > num_cols:
                cols = cols[:num_cols]
            elif len(cols) < num_cols:
                cols += [""] * (num_cols - len(cols))

            rows.append(cols)

        df = pd.DataFrame(rows, columns=headers)
        return df


    def get_ground_truth_map(self) -> dict:
        gt_map = {}
        for _, row in self.fields_df.iterrows():
            if "GT" in row and not pd.isna(row["GT"]):
                gt_map[row["Field Name"]] = str(row["GT"]).strip()
        return gt_map

    def save_to_excel(self, df: pd.DataFrame, output_path: str = "extracted_fields.xlsx"):
        df.to_excel(output_path, index=False)
        print(f"Excel file saved successfully at {output_path}")

# if __name__=="__main__":
#     pdf_path='acd.pdf'
#     excel_path='fields.xlsx'
#     extractor = FieldExtractor(pdf_path, excel_path, '')
#     df = extractor.extract_fields()
#     extractor.save_to_excel(df)