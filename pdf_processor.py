import fitz  # PyMuPDF
import io
import base64
from PIL import Image
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Processes PDF documents to extract text and images."""
    
    def __init__(self):
        self.supported_image_formats = ['png', 'jpeg', 'jpg']
    
    def process_pdf(self, pdf_path: str) -> Dict:
        """
        Process a PDF file and extract text and images.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text and images
        """
        try:
            doc = fitz.open(pdf_path)
            extracted_data = {
                'text_chunks': [],
                'images': [],
                'metadata': {
                    'total_pages': len(doc),
                    'filename': pdf_path.split('/')[-1]
                }
            }
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text
                text = page.get_text()
                if text.strip():
                    extracted_data['text_chunks'].append({
                        'content': text.strip(),
                        'page': page_num + 1,
                        'type': 'text'
                    })
                
                # Extract images
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Convert to PIL Image
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            img_pil = Image.open(io.BytesIO(img_data))
                            
                            # Convert to base64 for storage
                            buffered = io.BytesIO()
                            img_pil.save(buffered, format="PNG")
                            img_base64 = base64.b64encode(buffered.getvalue()).decode()
                            
                            extracted_data['images'].append({
                                'content': img_base64,
                                'page': page_num + 1,
                                'image_index': img_index,
                                'type': 'image',
                                'format': 'png'
                            })
                        
                        pix = None  # Free memory
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")
                        continue
            
            doc.close()
            logger.info(f"Processed PDF: {extracted_data['metadata']['total_pages']} pages, "
                       f"{len(extracted_data['text_chunks'])} text chunks, "
                       f"{len(extracted_data['images'])} images")
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
    
    def extract_text_blocks(self, pdf_path: str) -> List[Dict]:
        """
        Extract text blocks with better structure preservation.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of text blocks with metadata
        """
        try:
            doc = fitz.open(pdf_path)
            text_blocks = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")
                
                for block in blocks["blocks"]:
                    if "lines" in block:  # Text block
                        block_text = ""
                        for line in block["lines"]:
                            for span in line["spans"]:
                                block_text += span["text"]
                            block_text += "\n"
                        
                        if block_text.strip():
                            text_blocks.append({
                                'content': block_text.strip(),
                                'page': page_num + 1,
                                'bbox': block["bbox"],
                                'type': 'text_block'
                            })
            
            doc.close()
            return text_blocks
            
        except Exception as e:
            logger.error(f"Error extracting text blocks from {pdf_path}: {e}")
            raise
    
    def get_page_image(self, pdf_path: str, page_num: int, dpi: int = 150) -> str:
        """
        Convert a PDF page to an image.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number (0-indexed)
            dpi: Resolution for the image
            
        Returns:
            Base64 encoded image
        """
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            
            # Render page as image
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to base64
            img_base64 = base64.b64encode(img_data).decode()
            
            doc.close()
            return img_base64
            
        except Exception as e:
            logger.error(f"Error converting page {page_num} to image: {e}")
            raise
