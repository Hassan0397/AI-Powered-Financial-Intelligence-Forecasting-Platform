# 📄 app/report_generator.py
# Final Step: Professional PDF Report Generation

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import json

print("📄 STARTING PROFESSIONAL PDF REPORT GENERATION")
print("=" * 60)

class FinancialReportGenerator:
    """Generates professional PDF financial reports with AI insights"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        self.report_data = {}
        
    def setup_custom_styles(self):
        """Setup custom styles for professional report formatting"""
        # Title style - Professional centered title
        if 'MainTitle' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='MainTitle',
                parent=self.styles['Heading1'],
                fontSize=26,
                textColor=colors.HexColor('#2C3E50'),
                spaceAfter=24,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            ))
        
        # Section header style - Clean with subtle background
        if 'SectionHeader' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='SectionHeader',
                parent=self.styles['Heading2'],
                fontSize=18,
                textColor=colors.HexColor('#FFFFFF'),
                spaceAfter=12,
                spaceBefore=20,
                alignment=TA_LEFT,
                fontName='Helvetica-Bold',
                leftIndent=10,
                backColor=colors.HexColor('#34495E'),
                borderPadding=8,
                borderRadius=4
            ))
        
        # Subsection style - Professional with accent color
        if 'SubsectionHeader' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='SubsectionHeader',
                parent=self.styles['Heading3'],
                fontSize=14,
                textColor=colors.HexColor('#2980B9'),
                spaceAfter=8,
                spaceBefore=16,
                alignment=TA_LEFT,
                fontName='Helvetica-Bold',
                leftIndent=5
            ))
        
        # Body text style - Professional justified text
        if 'CustomBodyText' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='CustomBodyText',
                parent=self.styles['BodyText'],
                fontSize=11,
                textColor=colors.HexColor('#2C3E50'),
                spaceAfter=8,
                alignment=TA_JUSTIFY,
                fontName='Helvetica',
                leading=14
            ))
        
        # KPI style - Highlighted metrics
        if 'KPIText' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='KPIText',
                parent=self.styles['BodyText'],
                fontSize=12,
                textColor=colors.HexColor('#27AE60'),
                spaceAfter=4,
                alignment=TA_LEFT,
                fontName='Helvetica-Bold',
                backColor=colors.HexColor('#F8F9F9'),
                borderPadding=5,
                borderRadius=3
            ))
        
        # Table header style
        if 'TableHeader' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='TableHeader',
                parent=self.styles['BodyText'],
                fontSize=10,
                textColor=colors.white,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold',
                backColor=colors.HexColor('#2C3E50')
            ))
        
        # Footer style
        if 'FooterText' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='FooterText',
                parent=self.styles['BodyText'],
                fontSize=8,
                textColor=colors.HexColor('#7F8C8D'),
                alignment=TA_CENTER,
                fontName='Helvetica-Oblique'
            ))

    def load_data(self):
        """Load all required data for the report"""
        print("📁 Loading data for report generation...")
        
        try:
            # Define base directory
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, "data")
            outputs_dir = os.path.join(base_dir, "outputs")
            visuals_dir = os.path.join(outputs_dir, "visuals")
            
            print(f"📂 Looking for data in: {data_dir}")
            
            # Load cleaned financial data
            possible_data_paths = [
                os.path.join(data_dir, "financial_data_cleaned.csv"),
                os.path.join(data_dir, "cleaned_financial_data.csv"),
                os.path.join(base_dir, "financial_data_cleaned.csv"),
                "financial_data_cleaned.csv"
            ]
            
            self.df_cleaned = None
            for data_path in possible_data_paths:
                if os.path.exists(data_path):
                    self.df_cleaned = pd.read_csv(data_path)
                    print(f"✅ Loaded financial data from: {data_path}")
                    break
            
            if self.df_cleaned is None:
                print("❌ Could not find cleaned financial data file")
                return False
            
            # Convert date column if exists
            if 'Date' in self.df_cleaned.columns:
                self.df_cleaned['Date'] = pd.to_datetime(self.df_cleaned['Date'])
            
            # Load forecast data
            possible_forecast_paths = [
                os.path.join(data_dir, "financial_forecast.csv"),
                os.path.join(outputs_dir, "financial_forecast.csv"),
                os.path.join(base_dir, "financial_forecast.csv"),
                "financial_forecast.csv"
            ]
            
            self.df_forecast = pd.DataFrame()
            for forecast_path in possible_forecast_paths:
                if os.path.exists(forecast_path):
                    self.df_forecast = pd.read_csv(forecast_path)
                    print(f"✅ Loaded forecast data from: {forecast_path}")
                    if 'Date' in self.df_forecast.columns:
                        self.df_forecast['Date'] = pd.to_datetime(self.df_forecast['Date'])
                    break
            
            # Load AI-generated content
            possible_ai_paths = [
                os.path.join(outputs_dir, "ai_financial_report.json"),
                os.path.join(base_dir, "ai_financial_report.json"),
                "ai_financial_report.json"
            ]
            
            self.ai_report = {}
            for ai_path in possible_ai_paths:
                if os.path.exists(ai_path):
                    with open(ai_path, "r") as f:
                        self.ai_report = json.load(f)
                    print(f"✅ Loaded AI financial report from: {ai_path}")
                    break
            
            if not self.ai_report:
                print("⚠️  No AI report found, using placeholder content")
                self.ai_report = {
                    'executive_summary': 'AI analysis content not available. Please run the AI analysis step first.',
                    'forecast_analysis': 'Forecast analysis not available. Please run the forecasting step first.'
                }
            
            # Check for visualizations
            self.visuals_dir = visuals_dir
            self.available_visuals = []
            if os.path.exists(self.visuals_dir):
                self.available_visuals = [f for f in os.listdir(self.visuals_dir) 
                                        if f.endswith(('.png', '.jpg', '.jpeg'))]
                print(f"✅ Found {len(self.available_visuals)} visualization files")
            else:
                print("⚠️  No visuals directory found")
            
            # Store directories for later use
            self.data_dir = data_dir
            self.outputs_dir = outputs_dir
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def calculate_kpis(self):
        """Calculate key performance indicators for the report"""
        print("📊 Calculating KPIs...")
        
        self.kpis = {}
        
        # Basic financial metrics
        self.kpis['Total Sales'] = f"${self.df_cleaned['Sales'].sum():,.2f}"
        self.kpis['Total Profit'] = f"${self.df_cleaned['Profit'].sum():,.2f}"
        self.kpis['Average Sales'] = f"${self.df_cleaned['Sales'].mean():,.2f}"
        self.kpis['Average Profit'] = f"${self.df_cleaned['Profit'].mean():,.2f}"
        
        # Profit margin
        if 'Profit_Margin' in self.df_cleaned.columns:
            avg_margin = self.df_cleaned['Profit_Margin'].mean()
        else:
            avg_margin = (self.df_cleaned['Profit'].sum() / self.df_cleaned['Sales'].sum()) * 100
        self.kpis['Profit Margin'] = f"{avg_margin:.2f}%"
        
        # Performance leaders
        segment_perf = self.df_cleaned.groupby('Segment')['Profit'].sum()
        country_perf = self.df_cleaned.groupby('Country')['Profit'].sum()
        product_perf = self.df_cleaned.groupby('Product')['Profit'].sum()
        
        self.kpis['Top Segment'] = f"{segment_perf.idxmax()}"
        self.kpis['Top Country'] = f"{country_perf.idxmax()}"
        self.kpis['Top Product'] = f"{product_perf.idxmax()}"
        
        # Date range and transactions
        self.kpis['Analysis Period'] = f"{self.df_cleaned['Date'].min().strftime('%b %Y')} - {self.df_cleaned['Date'].max().strftime('%b %Y')}"
        self.kpis['Total Transactions'] = f"{len(self.df_cleaned):,}"
        
        print("✅ KPIs calculated successfully")

    def create_cover_page(self):
        """Create professional cover page"""
        print("🖼️  Creating cover page...")
        
        elements = []
        
        # Add main title
        title = Paragraph("FINANCIAL PERFORMANCE REPORT", self.styles['MainTitle'])
        elements.append(Spacer(1, 2.5*inch))
        elements.append(title)
        elements.append(Spacer(1, 0.3*inch))
        
        # Add subtitle
        subtitle = Paragraph("AI-Powered Business Intelligence Analysis", self.styles['Heading2'])
        subtitle.alignment = TA_CENTER
        elements.append(subtitle)
        elements.append(Spacer(1, 1.5*inch))
        
        # Add report details in a clean box
        details_style = ParagraphStyle(
            name='CoverDetails',
            parent=self.styles['BodyText'],
            fontSize=12,
            alignment=TA_CENTER,
            spaceAfter=12,
            leftIndent=0,
            rightIndent=0,
            textColor=colors.HexColor('#2C3E50')
        )
        
        details = [
            f"Generated on: {datetime.now().strftime('%B %d, %Y')}",
            f"Data Period: {self.kpis['Analysis Period']}",
            f"Total Records Analyzed: {self.kpis['Total Transactions']}",
            "",
            "Confidential - For Internal Use Only"
        ]
        
        for detail in details:
            if detail.strip():
                para = Paragraph(detail, details_style)
                elements.append(para)
        
        elements.append(Spacer(1, 2*inch))
        
        # Add methodology note
        method_text = "This report utilizes advanced analytics and AI to provide actionable business insights."
        method_para = Paragraph(method_text, self.styles['CustomBodyText'])
        method_para.alignment = TA_CENTER
        elements.append(method_para)
        
        elements.append(PageBreak())
        return elements

    def create_executive_summary(self):
        """Create executive summary section"""
        print("📝 Creating executive summary...")
        
        elements = []
        
        # Section header
        section_title = Paragraph("EXECUTIVE SUMMARY", self.styles['SectionHeader'])
        elements.append(section_title)
        
        # Extract AI summary
        ai_summary = self.ai_report.get('executive_summary', 'Comprehensive financial analysis completed.')
        
        # Clean and format the summary
        paragraphs = [p.strip() for p in ai_summary.split('\n\n') if p.strip()]
        
        for para in paragraphs[:4]:  # Limit to first 4 substantial paragraphs
            if len(para) > 100:
                para_elem = Paragraph(para, self.styles['CustomBodyText'])
                elements.append(para_elem)
                elements.append(Spacer(1, 0.15*inch))
        
        elements.append(Spacer(1, 0.3*inch))
        
        # Key metrics highlight box
        highlights_title = Paragraph("Performance Highlights", self.styles['SubsectionHeader'])
        elements.append(highlights_title)
        
        # Create a professional metrics table
        highlight_data = [
            ['METRIC', 'VALUE'],
            ['Total Sales', self.kpis['Total Sales']],
            ['Total Profit', self.kpis['Total Profit']],
            ['Profit Margin', self.kpis['Profit Margin']],
            ['Top Segment', self.kpis['Top Segment']],
            ['Analysis Period', self.kpis['Analysis Period']]
        ]
        
        highlights_table = Table(highlight_data, colWidths=[2.5*inch, 2.5*inch])
        highlights_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C3E50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9F9')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#2C3E50')),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#BDC3C7')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F2F4F4')])
        ]))
        
        elements.append(highlights_table)
        elements.append(PageBreak())
        
        return elements

    def create_financial_overview(self):
        """Create detailed financial overview"""
        print("💰 Creating financial overview...")
        
        elements = []
        
        section_title = Paragraph("FINANCIAL PERFORMANCE OVERVIEW", self.styles['SectionHeader'])
        elements.append(section_title)
        
        # Comprehensive KPI table
        kpi_data = [['KEY PERFORMANCE INDICATOR', 'VALUE']]
        
        # Group KPIs logically
        financial_metrics = ['Total Sales', 'Total Profit', 'Average Sales', 'Average Profit', 'Profit Margin']
        performance_metrics = ['Top Segment', 'Top Country', 'Top Product', 'Analysis Period', 'Total Transactions']
        
        kpi_data.extend([[kpi, self.kpis[kpi]] for kpi in financial_metrics])
        kpi_data.append(['', ''])  # Spacer row
        kpi_data.extend([[kpi, self.kpis[kpi]] for kpi in performance_metrics])
        
        kpi_table = Table(kpi_data, colWidths=[3.2*inch, 2.3*inch])
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495E')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, 6), colors.HexColor('#FFFFFF')),
            ('BACKGROUND', (0, 8), (-1, -1), colors.HexColor('#FFFFFF')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#2C3E50')),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D5DBDB')),
            ('LINEBELOW', (0, 6), (-1, 6), 1, colors.HexColor('#E5E8E8'))
        ]))
        
        elements.append(kpi_table)
        elements.append(Spacer(1, 0.4*inch))
        
        return elements

    def create_visual_analysis(self):
        """Create visual analysis section with professional layout"""
        print("📊 Adding visual analysis...")
        
        elements = []
        
        section_title = Paragraph("VISUAL ANALYSIS & TRENDS", self.styles['SectionHeader'])
        elements.append(section_title)
        
        if not self.available_visuals:
            no_viz_text = Paragraph("No visualizations available. Please generate charts in previous steps.", self.styles['CustomBodyText'])
            elements.append(no_viz_text)
            elements.append(PageBreak())
            return elements
        
        # Add visualizations in a clean grid layout
        for i, viz_file in enumerate(self.available_visuals[:4]):
            viz_path = os.path.join(self.visuals_dir, viz_file)
            
            if os.path.exists(viz_path):
                # Add visualization title
                viz_name = viz_file.replace('.png', '').replace('_', ' ').title()
                viz_title = Paragraph(f"Figure {i+1}: {viz_name}", self.styles['SubsectionHeader'])
                elements.append(viz_title)
                
                # Add the image with consistent sizing
                try:
                    img = Image(viz_path, width=6*inch, height=3.5*inch)
                    img.hAlign = 'CENTER'
                    elements.append(img)
                    elements.append(Spacer(1, 0.15*inch))
                    
                    # Add professional caption
                    caption_text = self.get_chart_caption(viz_file)
                    caption = Paragraph(caption_text, self.styles['CustomBodyText'])
                    caption.alignment = TA_CENTER
                    elements.append(caption)
                    elements.append(Spacer(1, 0.3*inch))
                    
                except Exception as e:
                    error_msg = Paragraph(f"Visualization not available: {viz_file}", self.styles['CustomBodyText'])
                    elements.append(error_msg)
        
        elements.append(PageBreak())
        return elements
    
    def get_chart_caption(self, viz_file):
        """Generate professional chart captions"""
        captions = {
            'sales_trend': "Historical sales performance showing trends, seasonality, and growth patterns over time.",
            'profit_by_segment': "Profitability analysis across business segments identifying high-performing categories.",
            'correlation_heatmap': "Correlation analysis between key business metrics revealing important relationships.",
            'top_products_profit': "Product performance ranking based on profitability and contribution margins.",
            'forecast': "Future performance projections based on historical trends and predictive modeling."
        }
        
        for key, caption in captions.items():
            if key in viz_file.lower():
                return caption
        
        return "Business intelligence visualization providing key performance insights."

    def create_forecast_section(self):
        """Create professional forecasting section"""
        print("🔮 Creating forecast section...")
        
        elements = []
        
        section_title = Paragraph("FINANCIAL FORECAST & PROJECTIONS", self.styles['SectionHeader'])
        elements.append(section_title)
        
        # Forecast summary
        forecast_intro = Paragraph("Future Outlook", self.styles['SubsectionHeader'])
        elements.append(forecast_intro)
        
        forecast_analysis = self.ai_report.get('forecast_analysis', 'Forecast analysis provides future performance projections.')
        
        # Format forecast text
        forecast_paragraphs = [p.strip() for p in forecast_analysis.split('\n\n') if p.strip()]
        for para in forecast_paragraphs[:2]:
            if len(para) > 50:
                para_elem = Paragraph(para, self.styles['CustomBodyText'])
                elements.append(para_elem)
                elements.append(Spacer(1, 0.1*inch))
        
        elements.append(Spacer(1, 0.2*inch))
        
        # Forecast data table
        if not self.df_forecast.empty and 'Predicted_Sales' in self.df_forecast.columns:
            forecast_table_title = Paragraph("Forecast Data", self.styles['SubsectionHeader'])
            elements.append(forecast_table_title)
            
            # Prepare professional forecast table
            forecast_data = [['PERIOD', 'PREDICTED SALES', 'PREDICTED PROFIT']]
            for _, row in self.df_forecast.head(6).iterrows():  # Limit to 6 periods
                date_str = row['Date'].strftime('%b %Y') if hasattr(row['Date'], 'strftime') else str(row['Date'])
                sales = f"${row.get('Predicted_Sales', 0):,.0f}" 
                profit = f"${row.get('Predicted_Profit', 0):,.0f}"
                forecast_data.append([date_str, sales, profit])
            
            forecast_table = Table(forecast_data, colWidths=[1.8*inch, 1.8*inch, 1.8*inch])
            forecast_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2980B9')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#EBF5FB')),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#2C3E50')),
                ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#AED6F1')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F4F8FB')])
            ]))
            
            elements.append(forecast_table)
        
        elements.append(Spacer(1, 0.3*inch))
        return elements

    def create_recommendations_section(self):
        """Create professional recommendations section"""
        print("💡 Creating recommendations section...")
        
        elements = []
        
        section_title = Paragraph("STRATEGIC RECOMMENDATIONS", self.styles['SectionHeader'])
        elements.append(section_title)
        
        # Recommendations header
        rec_intro = Paragraph("Actionable Insights", self.styles['SubsectionHeader'])
        elements.append(rec_intro)
        
        # Extract or create recommendations
        ai_content = self.ai_report.get('executive_summary', '')
        recommendations_text = """
        • Optimize resource allocation towards high-performing segments and products
        • Implement targeted marketing strategies for top-performing geographic markets
        • Enhance cost control measures to maintain and improve profit margins
        • Leverage predictive insights for strategic planning and inventory management
        • Develop data-driven decision making processes across the organization
        • Focus on customer segments demonstrating highest profitability and growth potential
        """
        
        # Format recommendations as bullet points
        rec_lines = [line.strip() for line in recommendations_text.split('\n') if line.strip() and '•' in line]
        
        for line in rec_lines:
            bullet_para = Paragraph(f"• {line[1:].strip()}", self.styles['CustomBodyText'])
            elements.append(bullet_para)
            elements.append(Spacer(1, 0.08*inch))
        
        elements.append(Spacer(1, 0.3*inch))
        
        # Conclusion
        conclusion_title = Paragraph("Conclusion", self.styles['SubsectionHeader'])
        elements.append(conclusion_title)
        
        conclusion_text = """
        This comprehensive financial analysis provides valuable insights into business performance and future opportunities. 
        The integration of advanced analytics with artificial intelligence enables data-driven decision making and supports 
        sustainable business growth. Regular monitoring of these metrics and continuous refinement of analytical models 
        will ensure ongoing optimization of financial performance and strategic alignment with organizational goals.
        """
        
        conclusion_para = Paragraph(conclusion_text, self.styles['CustomBodyText'])
        elements.append(conclusion_para)
        
        return elements

    def create_footer(self, canvas, doc):
        """Create professional footer"""
        canvas.saveState()
        
        # Footer text
        footer_style = self.styles['FooterText']
        footer_text = f"AI Financial Report | Page {doc.page} | {datetime.now().strftime('%Y-%m-%d')} | Confidential"
        
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#7F8C8D'))
        canvas.drawString(72, 50, footer_text)
        
        canvas.restoreState()

    def generate_report(self):
        """Main method to generate the complete PDF report"""
        print("🚀 Generating complete PDF report...")
        
        # Load data
        if not self.load_data():
            print("❌ Failed to load data. Report generation aborted.")
            return False
        
        # Calculate KPIs
        self.calculate_kpis()
        
        # Create PDF document with professional margins
        output_path = os.path.join(self.outputs_dir, "financial_report.pdf")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=54,    # Tighter margins for more content space
            leftMargin=54,
            topMargin=72,
            bottomMargin=72
        )
        
        # Build story (content elements)
        story = []
        
        # Add all sections in professional order
        story.extend(self.create_cover_page())
        story.extend(self.create_executive_summary())
        story.extend(self.create_financial_overview())
        story.extend(self.create_visual_analysis())
        story.extend(self.create_forecast_section())
        story.extend(self.create_recommendations_section())
        
        # Build PDF
        try:
            doc.build(story, onFirstPage=self.create_footer, onLaterPages=self.create_footer)
            print(f"✅ PDF report successfully generated: {output_path}")
            
            # Verify file creation
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / 1024 / 1024
                print(f"📁 Report file size: {file_size:.2f} MB")
                print("🎨 Report features: Professional formatting, perfect alignment, clean layout")
                return True
            else:
                print("❌ Report file was not created")
                return False
                
        except Exception as e:
            print(f"❌ Error generating PDF: {e}")
            return False

# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 AI-POWERED FINANCIAL REPORT GENERATOR")
    print("=" * 60)
    
    # Initialize generator
    generator = FinancialReportGenerator()
    
    # Generate report
    success = generator.generate_report()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 PROFESSIONAL FINANCIAL REPORT COMPLETED!")
        print("=" * 60)
        print("""
        ✅ PROFESSIONAL FEATURES INCLUDED:
        
        🎨 DESIGN & FORMATTING:
          • Clean, professional layout with perfect alignment
          • Consistent color scheme and typography
          • Professional tables with alternating row colors
          • Centered images with descriptive captions
          • Proper spacing and margins throughout
        
        📊 CONTENT STRUCTURE:
          • Executive Summary with key highlights
          • Comprehensive Financial Overview
          • Visual Analysis with professional charts
          • Forecasting with data tables
          • Strategic Recommendations
        
        📄 PROFESSIONAL ELEMENTS:
          • Cover page with company branding
          • Section headers with background colors
          • Justified text for better readability
          • Consistent font sizes and styles
          • Professional footer on every page
        
        🎯 READY FOR EXECUTIVE PRESENTATION!
        """)
    else:
        print("\n❌ Report generation failed. Please check the error messages above.")