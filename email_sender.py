"""
Email notification system for portfolio reports
Sends formatted HTML emails with portfolio analysis and recommendations
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import logging
from datetime import datetime
from typing import Dict, Optional
import os

logger = logging.getLogger(__name__)

class EmailSender:
    """Handles email notifications for portfolio reports"""
    
    def __init__(self, email_config: dict):
        self.config = email_config
        
    def send_portfolio_report(self, report_content: str, risk_chart_path: str = None, csv_path: str = None) -> bool:
        """
        Send weekly portfolio report via email, with optional risk chart and CSV attachments.
        Returns True if successful, False otherwise
        """
        logger.info("Sending portfolio report email")
        try:
            # Create message
            msg = MIMEMultipart('mixed')
            msg['Subject'] = f"Weekly Portfolio Review - {datetime.now().strftime('%B %d, %Y')}"
            msg['From'] = self.config['from_email']
            msg['To'] = self.config['to_email']

            # Create both plain text and HTML versions
            text_content = self._convert_to_text(report_content)
            html_content = self._convert_to_html(report_content)

            # Create MIMEText objects
            part1 = MIMEText(text_content, 'plain')
            part2 = MIMEText(html_content, 'html')

            # Attach text and HTML parts
            alt_part = MIMEMultipart('alternative')
            alt_part.attach(part1)
            alt_part.attach(part2)
            msg.attach(alt_part)

            # Attach risk chart if provided
            if risk_chart_path and os.path.exists(risk_chart_path):
                with open(risk_chart_path, 'rb') as f:
                    mime = MIMEBase('image', 'png')
                    mime.set_payload(f.read())
                    encoders.encode_base64(mime)
                    mime.add_header('Content-Disposition', 'attachment', filename=os.path.basename(risk_chart_path))
                    msg.attach(mime)
                logger.info(f"Attached risk chart: {risk_chart_path}")

            # Attach CSV if provided
            if csv_path and os.path.exists(csv_path):
                with open(csv_path, 'rb') as f:
                    mime = MIMEBase('application', 'octet-stream')
                    mime.set_payload(f.read())
                    encoders.encode_base64(mime)
                    mime.add_header('Content-Disposition', 'attachment', filename=os.path.basename(csv_path))
                    msg.attach(mime)
                logger.info(f"Attached CSV: {csv_path}")

            # Send email
            success = self._send_email(msg)

            if success:
                logger.info("Portfolio report email sent successfully (status: delivered)")
            else:
                logger.error("Failed to send portfolio report email (status: failed)")

            return success

        except Exception as e:
            logger.error(f"Email sending failed: {e}")
            return False
    
    def send_alert_email(self, subject: str, message: str, priority: str = 'normal') -> bool:
        """
        Send alert email for urgent portfolio matters
        """
        logger.info(f"Sending alert email: {subject}")
        
        try:
            msg = MIMEMultipart()
            msg['Subject'] = f"Portfolio Alert: {subject}"
            msg['From'] = self.config['from_email']
            msg['To'] = self.config['to_email']
            
            if priority == 'high':
                msg['X-Priority'] = '1'
                msg['X-MSMail-Priority'] = 'High'
            
            # Create message body
            body = f"""
Portfolio Alert
===============

{message}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
Automated Portfolio Management System
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            return self._send_email(msg)
            
        except Exception as e:
            logger.error(f"Alert email sending failed: {e}")
            return False
    
    def _send_email(self, msg: MIMEMultipart) -> bool:
        """Send email using SMTP"""
        try:
            with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                server.starttls()
                server.login(self.config['username'], self.config['password'])
                server.send_message(msg)
                
            return True
            
        except Exception as e:
            logger.error(f"SMTP sending failed: {e}")
            return False
    
    def _convert_to_text(self, markdown_content: str) -> str:
        """Convert markdown to plain text for email"""
        # Simple markdown to text conversion
        text = markdown_content
        
        # Remove markdown headers
        text = text.replace('# ', '').replace('## ', '').replace('### ', '')
        
        # Remove markdown formatting
        text = text.replace('**', '').replace('*', '')
        text = text.replace('⚠️', 'WARNING: ')
        
        return text
    
    def _convert_to_html(self, markdown_content: str) -> str:
        """Convert markdown to HTML for email"""
        # Simple markdown to HTML conversion
        html = markdown_content
        
        # Convert headers
        html = html.replace('# ', '<h1>').replace('\n# ', '</h1>\n<h1>')
        html = html.replace('## ', '<h2>').replace('\n## ', '</h2>\n<h2>')
        html = html.replace('### ', '<h3>').replace('\n### ', '</h3>\n<h3>')
        
        # Convert bold text
        html = html.replace('**', '<strong>').replace('**', '</strong>')
        
        # Convert bullet points
        html = html.replace('• ', '<li>').replace('\n• ', '</li>\n<li>')
        
        # Add HTML structure
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; }}
        h2 {{ color: #34495e; }}
        h3 {{ color: #7f8c8d; }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        .warning {{ color: #f39c12; }}
        li {{ margin: 5px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    {html}
</body>
</html>
"""
        
        return html_template
    
    def test_email_connection(self) -> bool:
        """Test email configuration"""
        logger.info("Testing email connection")
        
        try:
            test_subject = "Portfolio System - Email Test"
            test_message = "This is a test email from your automated portfolio management system."
            
            return self.send_alert_email(test_subject, test_message)
            
        except Exception as e:
            logger.error(f"Email connection test failed: {e}")
            return False
