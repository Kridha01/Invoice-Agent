import streamlit as st
import os
import pandas as pd
from io import BytesIO
from rag import FieldExtractor

st.set_page_config(
    page_title="Business Invoice Processor",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Business Invoice Processor")
st.write(
    "Upload your invoice PDF and a fields Excel file to extract structured data."
)

sample_df = pd.DataFrame({
    "Field Name": [
        "Invoice Number",
        "Customer Name",
        "Customer Address",
        "Invoice Date",
        "VAT Number",
        "Total Amount",
        "VAT Amount",
        "Gross Amount"
    ],
    "Description": [
        "Unique invoice identifier",
        "Name of the customer",
        "Full billing address",
        "Date when invoice was issued",
        "VAT registration number",
        "Net payable invoice amount",
        "Value added tax amount",
        "Total amount including tax"
    ],
    "ground_truth": [
        "123100401",
        "Mr. John Doe",
        "Musterstr. 23, 12345 Musterstadt",
        "01.02.2024",
        "DE199378386",
        "381.12",
        "72.41",
        "453.53"
    ]
})

buffer = BytesIO()
sample_df.to_excel(buffer, index=False)
buffer.seek(0)

st.download_button(
    label="📥 Download Sample Fields Excel to be Uploaded",
    data=buffer,
    file_name="sample_fields_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.markdown("---")

pdf_file = st.file_uploader("Upload Invoice PDF", type=["pdf"])
excel_file = st.file_uploader("Upload Fields Excel", type=["xlsx", "xls"])

if st.button("Process"):
    if pdf_file is None or excel_file is None:
        st.error("Please upload both PDF and Excel files.")
    else:
        os.makedirs("temp", exist_ok=True)

        pdf_path = os.path.join("temp", pdf_file.name)
        excel_path = os.path.join("temp", excel_file.name)

        with open(pdf_path, "wb") as f:
            f.write(pdf_file.getbuffer())

        with open(excel_path, "wb") as f:
            f.write(excel_file.getbuffer())

        

        extractor = FieldExtractor(pdf_path, excel_path, '')
        df = extractor.extract_fields()

        output_path = os.path.join("temp", f"extracted_{pdf_file.name}.xlsx")
        extractor.save_to_excel(df, output_path)

        st.success("Processed Successfully")
        st.dataframe(df)

        with open(output_path, "rb") as f:
            st.download_button(
                "📥 Download Processed Excel",
                data=f,
                file_name=f"extracted_{pdf_file.name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # Cleanup uploaded files
        os.remove(pdf_path)
        os.remove(excel_path)

# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    "📧 Email: rohansinghquantam@gmail.com"
    "📞 Phone: +91-9876543210"
)
