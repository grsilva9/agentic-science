import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from schemas import ReportInput, ReportOutput
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Table, TableStyle, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
import torch
import math
from pathlib import Path
import matplotlib.pyplot as plt

#1. Defines report function.
def report_task(input: ReportInput) -> ReportOutput:
    """
    Generate a structured PDF report using Platypus with a dynamic image grid.
    
    Parameters:
    - input: Includes image path, table path, plot path, text data, and report name.
    
    Returns:
    - ReportOutput: The path to the generated PDF.
    """
    
    # Set up the document
    pdf_dir = Path("Reports")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / f"{input.report_name}.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    
    # Get the style sheet
    styles = getSampleStyleSheet()
    story = []

    # Title Section
    title = Paragraph("<b>Title of the Report</b>", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))  # Space between title and introduction

    # Introduction Section
    intro = Paragraph(input.text_data["introduction"], styles['Normal'])
    story.append(intro)
    story.append(Spacer(1, 12))

    # Image Section
    image_title = Paragraph("<b>Images from .pth file</b>", styles['Heading2'])
    story.append(image_title)
    image_desc = Paragraph(input.text_data["image_description"], styles['Normal'])
    story.append(image_desc)
    story.append(Spacer(1, 12))
    
    # Load .pth images (PyTorch tensor)
    images_tensor = torch.load(input.gen_images_path)
    images = images_tensor.numpy()  # Convert tensor to numpy
    images = np.transpose(images, (0, 2, 3, 1))  # Convert to HWC format for images
    
    # Plotting images from the .pth file (dynamic grid)
    num_images = len(images)
    max_columns = 4  # Max number of columns per row
    rows = math.ceil(num_images / max_columns)
    
    # Create a figure with a specific size
    figsize = (16, 12)
    fig, axes = plt.subplots(rows, max_columns, figsize=figsize)
    
    axes = axes.flatten()  # Flatten the axes array for easy iteration
    
    for i, ax in enumerate(axes):
        if i < num_images:
            img = images[i]
            # Ensure the image is in the correct shape (H, W, C) for RGB or (H, W) for grayscale
            if img.ndim == 3 and img.shape[2] == 1:  # If image has shape (H, W, 1), convert to (H, W)
                img = np.squeeze(img, axis=-1)  # Remove the last dimension
            
            # Ensure the image values are in the range [0, 255] for uint8
            img = np.clip(img * 255, 0, 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
            
            # If the image is grayscale (2D), convert to a 3-channel format (RGB) for consistency
            if img.ndim == 2:  # Grayscale image (H, W)
                img = np.stack([img] * 3, axis=-1)  # Convert to (H, W, 3)

            ax.imshow(img)
            ax.axis('off')
        else:
            ax.axis('off')  # Turn off the axis for extra empty subplots

    plt.tight_layout()  # Adjust the layout to avoid overlap
    image_grid_path = 'temp_images_grid.png'
    plt.savefig(image_grid_path, dpi=300)
    plt.close(fig)

    story.append(Image(image_grid_path, width=400, height=200))  # Add image grid to the story
    story.append(Spacer(1, 12))

    # Table Section
    table_title = Paragraph("<b>Data Table</b>", styles['Heading2'])
    story.append(table_title)
    
    table_desc = Paragraph(input.text_data["table_description"], styles['Normal'])
    story.append(table_desc)
    story.append(Spacer(1, 12))

    # Load and display the table from CSV
    table_data = pd.read_csv(input.stats_csv)
    table_data_rows = [list(row) for row in table_data.values]
    table = Table(table_data_rows)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    # Plot Section
    plot_title = Paragraph("<b>Plot Image</b>", styles['Heading2'])
    story.append(plot_title)
    
    plot_desc = Paragraph(input.text_data["plot_description"], styles['Normal'])
    story.append(plot_desc)
    story.append(Spacer(1, 12))
    
    # Adding the plot image
    plot_image = Image(str(input.plot_png), width=400, height=200)
    story.append(plot_image)
    
    # Build the document
    try:
        doc.build(story)
        print(f"Report saved to {str(pdf_path)}")
    except Exception as e:
        print(f"Error while generating PDF: {e}")
    
    return ReportOutput(report_path=str(pdf_path))
