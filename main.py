#!/usr/bin/env python3
"""
Automated Portfolio Management System
Runs every Monday at 7:30 AM to generate portfolio reports and rebalancing recommendations
"""

import schedule
import time
import logging
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from portfolio_manager import PortfolioManager
from config import PORTFOLIO_CONFIG, EMAIL_CONFIG, update_portfolio_prices, PORTFOLIO_HOLDINGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('portfolio_automation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def run_weekly_portfolio_review():
    """
    Main function that runs the complete portfolio review process
    This executes every Monday at 7:30 AM
    """
    try:
        logger.info("Starting weekly portfolio review...")
        
        # Update prices before risk analysis
        update_portfolio_prices()
        
        # Initialize portfolio manager
        portfolio = PortfolioManager(PORTFOLIO_CONFIG)
        
        # Execute the complete analysis workflow
        report = portfolio.generate_weekly_report()
        
        # Send email notification
        portfolio.send_email_report(report)
        
        logger.info("Weekly portfolio review completed successfully")
        
    except Exception as e:
        logger.error(f"Error in portfolio review: {str(e)}")
        # Send error notification email
        send_error_notification(str(e))

def send_error_notification(error_message):
    """Send email notification when automation fails"""
    try:
        msg = MIMEText(f"Portfolio automation failed with error: {error_message}")
        msg['Subject'] = "Portfolio Automation Error"
        msg['From'] = EMAIL_CONFIG['from_email']
        msg['To'] = EMAIL_CONFIG['to_email']
        
        with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
            server.starttls()
            server.login(EMAIL_CONFIG['username'], EMAIL_CONFIG['password'])
            server.send_message(msg)
            
    except Exception as e:
        logger.error(f"Failed to send error notification: {str(e)}")

# Schedule the weekly review for Monday at 7:30 AM
schedule.every().monday.at("07:30").do(run_weekly_portfolio_review)

if __name__ == "__main__":
    logger.info("Portfolio automation system started")
    logger.info("Scheduled to run every Monday at 7:30 AM")
    
    # Run once immediately for testing (comment out in production)
    run_weekly_portfolio_review()
    
    # # Keep the script running and check for scheduled tasks
    # while True:
    #     schedule.run_pending()
    #     time.sleep(60)  # Check every minute
