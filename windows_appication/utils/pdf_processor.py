"""
PDF Processor Module

This module handles the processing of PDF files.
It extracts text from PDFs and provides functionality for summarization.
"""

import io
import re
import PyPDF2
from typing import BinaryIO, Union, Optional

class PDFProcessor:
    def __init__(self):
        """Initialize the PDF processor."""
        pass
    
    def extract_text(self, pdf_file: Union[BinaryIO, str]) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_file: The PDF file to extract text from (file-like object or path)
            
        Returns:
            str: The extracted text
        """
        try:
            # Handle file-like objects from Streamlit
            if hasattr(pdf_file, 'read') and callable(pdf_file.read):
                pdf_content = pdf_file.read()
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            # Handle file paths
            elif isinstance(pdf_file, str):
                pdf_reader = PyPDF2.PdfReader(pdf_file)
            else:
                raise ValueError("Invalid PDF file type. Expected file-like object or path.")
            
            # Extract text from all pages
            extracted_text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n\n"
            
            return self._clean_text(extracted_text)
        except Exception as e:
            return f"Error extracting text from PDF: {str(e)}"
    
    def _clean_text(self, text: str) -> str:
        """
        Clean the extracted text by removing extra whitespace and other artifacts.
        
        Args:
            text: The text to clean
            
        Returns:
            str: The cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive newlines and whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove page numbers (common in PDFs)
        text = re.sub(r'\n\s*\d+\s*\n', '\n\n', text)
        
        # Remove headers and footers (common in PDFs) - simple heuristic
        lines = text.split('\n')
        if len(lines) > 4:
            # Check if first and last lines are repeated on multiple pages
            first_line = lines[0].strip()
            last_line = lines[-1].strip()
            
            # Count occurrences
            first_count = sum(1 for line in lines if line.strip() == first_line)
            last_count = sum(1 for line in lines if line.strip() == last_line)
            
            # If they appear multiple times, they might be headers/footers
            if first_count > 1 and len(first_line) < 50:
                text = re.sub(f'^{re.escape(first_line)}$', '', text, flags=re.MULTILINE)
            if last_count > 1 and len(last_line) < 50:
                text = re.sub(f'^{re.escape(last_line)}$', '', text, flags=re.MULTILINE)
        
        # Clean up any remaining multi-spaces
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def extract_structured_data(self, pdf_file: Union[BinaryIO, str]) -> dict:
        """
        Extract structured data from a PDF file (form fields, tables, etc.)
        
        Args:
            pdf_file: The PDF file to extract data from
            
        Returns:
            dict: The extracted structured data
        """
        try:
            # This is a simplified version - real implementation would use libraries like tabula-py for tables
            # or pdfplumber for more complex extraction
            
            # Extract basic text
            text = self.extract_text(pdf_file)
            
            # Try to identify sections
            sections = {}
            
            # Simple section detection based on line starting with heading-like text
            section_pattern = r'^([A-Z][A-Za-z\s]{2,50}):(.+?)(?=^[A-Z][A-Za-z\s]{2,50}:|$)'
            found_sections = re.findall(section_pattern, text, re.MULTILINE | re.DOTALL)
            
            if found_sections:
                for section_name, section_content in found_sections:
                    sections[section_name.strip()] = section_content.strip()
            
            # Try to extract tables (simplified)
            tables = []
            # Real implementation would use tabula-py or similar library
            
            result = {
                "full_text": text,
                "sections": sections,
                "tables": tables
            }
            
            return result
        except Exception as e:
            return {"error": f"Error extracting structured data from PDF: {str(e)}"}
    
    def get_metadata(self, pdf_file: Union[BinaryIO, str]) -> dict:
        """
        Extract metadata from a PDF file.
        
        Args:
            pdf_file: The PDF file to extract metadata from
            
        Returns:
            dict: The extracted metadata
        """
        try:
            # Handle file-like objects from Streamlit
            if hasattr(pdf_file, 'read') and callable(pdf_file.read):
                pdf_content = pdf_file.read()
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            # Handle file paths
            elif isinstance(pdf_file, str):
                pdf_reader = PyPDF2.PdfReader(pdf_file)
            else:
                raise ValueError("Invalid PDF file type. Expected file-like object or path.")
            
            # Extract metadata
            info = pdf_reader.metadata
            if info:
                metadata = {
                    "Title": info.get("/Title", ""),
                    "Author": info.get("/Author", ""),
                    "Subject": info.get("/Subject", ""),
                    "Creator": info.get("/Creator", ""),
                    "Producer": info.get("/Producer", ""),
                    "CreationDate": info.get("/CreationDate", ""),
                    "ModificationDate": info.get("/ModDate", ""),
                }
            else:
                metadata = {"error": "No metadata found."}
            
            # Add basic document info
            metadata["Pages"] = len(pdf_reader.pages)
            
            return metadata
        except Exception as e:
            return {"error": f"Error extracting metadata from PDF: {str(e)}"}
