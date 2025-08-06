import fitz  # PyMuPDF
import io
import base64
from PIL import Image
from typing import List, Dict, Tuple, Optional
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Enhanced PDF processor that extracts text and images with spatial coordinates."""
    
    def __init__(self):
        self.supported_image_formats = ['png', 'jpeg', 'jpg']
        self.min_text_length = 10  # Minimum text length for chunks
        self.proximity_threshold = 50  # Pixels for spatial relationships
    
    def process_pdf(self, pdf_path: str) -> Dict:
        """
        Process a PDF file and extract text and images with spatial coordinates.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text, images, and spatial relationships
        """
        try:
            doc = fitz.open(pdf_path)
            extracted_data = {
                'text_chunks': [],
                'images': [],
                'spatial_relationships': [],
                'metadata': {
                    'total_pages': len(doc),
                    'filename': pdf_path.split('/')[-1]
                }
            }
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_data = self._process_page(page, page_num)
                
                # Add page data to extracted_data
                extracted_data['text_chunks'].extend(page_data['text_chunks'])
                extracted_data['images'].extend(page_data['images'])
                
                # Detect spatial relationships within the page
                relationships = self._detect_spatial_relationships(
                    page_data['text_chunks'], 
                    page_data['images']
                )
                extracted_data['spatial_relationships'].extend(relationships)
            
            doc.close()
            logger.info(f"Enhanced processing complete: {extracted_data['metadata']['total_pages']} pages, "
                       f"{len(extracted_data['text_chunks'])} text chunks, "
                       f"{len(extracted_data['images'])} images, "
                       f"{len(extracted_data['spatial_relationships'])} spatial relationships")
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
    
    def _process_page(self, page: fitz.Page, page_num: int) -> Dict:
        """Process a single page to extract text and images with coordinates."""
        page_data = {
            'text_chunks': [],
            'images': []
        }
        
        # Extract text with coordinates
        text_chunks = self._extract_text_with_coordinates(page, page_num)
        page_data['text_chunks'] = text_chunks
        
        # Extract images with coordinates
        images = self._extract_images_with_coordinates(page, page_num)
        page_data['images'] = images
        
        return page_data
    
    def _extract_text_with_coordinates(self, page: fitz.Page, page_num: int) -> List[Dict]:
        """Extract text chunks with bounding box coordinates."""
        text_chunks = []
        
        try:
            # Get structured text with coordinates
            blocks = page.get_text("dict")
            
            for block_idx, block in enumerate(blocks["blocks"]):
                if "lines" in block:  # Text block
                    block_text = ""
                    block_bbox = None
                    
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            if span["text"].strip():
                                line_text += span["text"]
                        
                        if line_text.strip():
                            block_text += line_text + " "
                            
                            # Update block bounding box
                            line_bbox = line["bbox"]
                            if block_bbox is None:
                                block_bbox = list(line_bbox)
                            else:
                                # Expand bounding box
                                block_bbox[0] = min(block_bbox[0], line_bbox[0])  # x0
                                block_bbox[1] = min(block_bbox[1], line_bbox[1])  # y0
                                block_bbox[2] = max(block_bbox[2], line_bbox[2])  # x1
                                block_bbox[3] = max(block_bbox[3], line_bbox[3])  # y1
                    
                    # Add text chunk if it meets minimum length
                    if block_text.strip() and len(block_text.strip()) >= self.min_text_length:
                        text_chunks.append({
                            'content': block_text.strip(),
                            'page': page_num + 1,
                            'bbox': block_bbox,
                            'block_index': block_idx,
                            'type': 'text',
                            'area': self._calculate_area(block_bbox) if block_bbox else 0
                        })
        
        except Exception as e:
            logger.warning(f"Failed to extract structured text from page {page_num + 1}: {e}")
            # Fallback to simple text extraction
            text = page.get_text()
            if text.strip():
                # Get page dimensions for fallback bbox
                rect = page.rect
                text_chunks.append({
                    'content': text.strip(),
                    'page': page_num + 1,
                    'bbox': [rect.x0, rect.y0, rect.x1, rect.y1],
                    'block_index': 0,
                    'type': 'text',
                    'area': self._calculate_area([rect.x0, rect.y0, rect.x1, rect.y1])
                })
        
        return text_chunks
    
    def _extract_images_with_coordinates(self, page: fitz.Page, page_num: int) -> List[Dict]:
        """Extract images with bounding box coordinates."""
        images = []
        
        try:
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get image reference
                    xref = img[0]
                    
                    # Get image bounding box
                    img_rects = page.get_image_rects(xref)
                    if not img_rects:
                        logger.warning(f"No rectangle found for image {img_index} on page {page_num + 1}")
                        continue
                    
                    # Use the first rectangle (images can appear multiple times)
                    img_rect = img_rects[0]
                    bbox = [img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1]
                    
                    # Extract image data
                    pix = fitz.Pixmap(page.parent, xref)
                    
                    # Convert to PIL Image
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        img_pil = Image.open(io.BytesIO(img_data))
                        
                        # Convert to base64 for storage
                        buffered = io.BytesIO()
                        img_pil.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()
                        
                        # Validate base64 data before storing
                        if img_base64 and len(img_base64) > 100:  # Ensure valid, non-empty base64
                            images.append({
                                'content': img_base64,
                                'page': page_num + 1,
                                'bbox': bbox,
                                'image_index': img_index,
                                'type': 'image',
                                'format': 'png',
                                'dimensions': [img_pil.width, img_pil.height],
                                'area': self._calculate_area(bbox)
                            })
                        else:
                            logger.warning(f"Skipping invalid/empty image data on page {page_num + 1}, image {img_index}")
                    
                    pix = None  # Free memory
                    
                except Exception as e:
                    logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")
                    continue
        
        except Exception as e:
            logger.warning(f"Failed to get images from page {page_num + 1}: {e}")
        
        return images
    
    def _detect_spatial_relationships(self, text_chunks: List[Dict], images: List[Dict]) -> List[Dict]:
        """Detect spatial relationships between text and images."""
        relationships = []
        
        for text_chunk in text_chunks:
            for image in images:
                # Only check relationships within the same page
                if text_chunk['page'] != image['page']:
                    continue
                
                relationship = self._analyze_spatial_relationship(text_chunk, image)
                if relationship:
                    relationships.append(relationship)
        
        return relationships
    
    def _analyze_spatial_relationship(self, text_chunk: Dict, image: Dict) -> Optional[Dict]:
        """Analyze spatial relationship between a text chunk and an image."""
        text_bbox = text_chunk['bbox']
        img_bbox = image['bbox']
        
        if not text_bbox or not img_bbox:
            return None
        
        # Calculate centers
        text_center = self._get_center(text_bbox)
        img_center = self._get_center(img_bbox)
        
        # Calculate distance
        distance = self._calculate_distance(text_center, img_center)
        
        # Determine relationship type
        relationship_type = self._determine_relationship_type(text_bbox, img_bbox)
        
        # Only return relationships that are close enough
        if distance <= self.proximity_threshold * 3:  # Extended threshold for relationships
            return {
                'text_chunk_id': f"text_{text_chunk['page']}_{text_chunk.get('block_index', 0)}",
                'image_id': f"img_{image['page']}_{image['image_index']}",
                'relationship_type': relationship_type,
                'distance': distance,
                'confidence': self._calculate_confidence(distance, relationship_type),
                'page': text_chunk['page']
            }
        
        return None
    
    def _determine_relationship_type(self, text_bbox: List[float], img_bbox: List[float]) -> str:
        """Determine the type of spatial relationship."""
        text_center = self._get_center(text_bbox)
        img_center = self._get_center(img_bbox)
        
        # Check if text is above image (potential caption above)
        if text_bbox[3] <= img_bbox[1] and abs(text_bbox[3] - img_bbox[1]) < self.proximity_threshold:
            return "caption_above"
        
        # Check if text is below image (potential caption below)
        if text_bbox[1] >= img_bbox[3] and abs(text_bbox[1] - img_bbox[3]) < self.proximity_threshold:
            return "caption_below"
        
        # Check if text is to the left of image
        if text_bbox[2] <= img_bbox[0] and abs(text_bbox[2] - img_bbox[0]) < self.proximity_threshold:
            return "adjacent_left"
        
        # Check if text is to the right of image
        if text_bbox[0] >= img_bbox[2] and abs(text_bbox[0] - img_bbox[2]) < self.proximity_threshold:
            return "adjacent_right"
        
        # Check for overlap
        if self._check_overlap(text_bbox, img_bbox):
            return "overlapping"
        
        # Default to nearby
        return "nearby"
    
    def _calculate_area(self, bbox: List[float]) -> float:
        """Calculate area of a bounding box."""
        if not bbox or len(bbox) != 4:
            return 0
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    def _get_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Get center point of a bounding box."""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _check_overlap(self, bbox1: List[float], bbox2: List[float]) -> bool:
        """Check if two bounding boxes overlap."""
        return not (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or 
                   bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1])
    
    def _calculate_confidence(self, distance: float, relationship_type: str) -> float:
        """Calculate confidence score for a spatial relationship."""
        # Base confidence on distance (closer = higher confidence)
        base_confidence = max(0, 1 - (distance / (self.proximity_threshold * 3)))
        
        # Adjust based on relationship type
        type_multipliers = {
            "caption_above": 1.2,
            "caption_below": 1.2,
            "adjacent_left": 1.0,
            "adjacent_right": 1.0,
            "overlapping": 0.8,  # Lower confidence for overlaps
            "nearby": 0.9
        }
        
        multiplier = type_multipliers.get(relationship_type, 1.0)
        return min(1.0, base_confidence * multiplier)
    
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
