#!/usr/bin/env python3
"""
Email Alert Module for Phoenix Rent Growth Forecasting

Purpose: Send email notifications for forecast updates and alerts
Usage:
    # Test email configuration
    python3 email_alerts.py --test

    # Send alerts programmatically
    from email_alerts import EmailAlerter
    alerter = EmailAlerter()
    alerter.send_forecast_alert(alerts, forecast_summary)

Configuration:
    config/email_config.json (copy from email_config.example.json)

Author: Generated for automated forecast system
Version: 1.0
"""

import smtplib
import json
import sys
import argparse
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from datetime import datetime
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Project paths
BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / 'config'
CONFIG_FILE = CONFIG_DIR / 'email_config.json'
EXAMPLE_CONFIG_FILE = CONFIG_DIR / 'email_config.example.json'

# ============================================================================
# EMAIL ALERTER CLASS
# ============================================================================

class EmailAlerter:
    """Handle email notifications for forecast system"""

    def __init__(self, config_file=None):
        """
        Initialize email alerter with configuration

        Args:
            config_file: Path to email configuration JSON file
        """
        self.config_file = config_file or CONFIG_FILE
        self.config = self._load_config()
        self.enabled = self.config.get('email_settings', {}).get('enabled', False)

    def _load_config(self):
        """Load email configuration from JSON file"""
        if not self.config_file.exists():
            print(f"Warning: Email config not found at {self.config_file}")
            print(f"Copy {EXAMPLE_CONFIG_FILE} to {CONFIG_FILE} and configure")
            return {}

        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading email config: {e}")
            return {}

    def is_enabled(self):
        """Check if email alerts are enabled"""
        return self.enabled

    def should_send(self, alerts, success=True):
        """
        Determine if email should be sent based on configuration and alerts

        Args:
            alerts: List of alert dictionaries
            success: Whether the forecast run was successful

        Returns:
            bool: True if email should be sent
        """
        if not self.enabled:
            return False

        settings = self.config.get('email_settings', {})
        threshold = settings.get('alert_threshold', {})

        # Check minimum alerts
        if len(alerts) < threshold.get('min_alerts_to_send', 1):
            return False

        # Check success scenario
        if success and not threshold.get('send_on_success', False):
            # Only send if there are warnings/critical alerts
            has_warnings = any(a.get('severity') in ['WARNING', 'CRITICAL'] for a in alerts)
            if not has_warnings:
                return False

        # Check severity thresholds
        has_critical = any(a.get('severity') == 'CRITICAL' for a in alerts)
        has_warnings = any(a.get('severity') == 'WARNING' for a in alerts)

        if has_critical and threshold.get('send_on_critical', True):
            return True

        if has_warnings and threshold.get('send_on_warnings', True):
            return True

        return False

    def _format_alert_html(self, alerts, forecast_summary=None):
        """
        Format alerts and summary as HTML email body

        Args:
            alerts: List of alert dictionaries
            forecast_summary: Optional forecast summary text

        Returns:
            str: HTML formatted email body
        """
        # Severity colors
        severity_colors = {
            'CRITICAL': '#dc3545',
            'WARNING': '#ffc107',
            'INFO': '#17a2b8'
        }

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
        .alert-section {{ margin: 20px; }}
        .alert {{ padding: 15px; margin: 10px 0; border-left: 4px solid; border-radius: 4px; }}
        .alert-critical {{ background-color: #f8d7da; border-color: #dc3545; }}
        .alert-warning {{ background-color: #fff3cd; border-color: #ffc107; }}
        .alert-info {{ background-color: #d1ecf1; border-color: #17a2b8; }}
        .summary {{ background-color: #f8f9fa; padding: 20px; margin: 20px; border-radius: 4px; }}
        .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; }}
        .severity-badge {{ display: inline-block; padding: 3px 8px; border-radius: 3px; color: white; font-weight: bold; font-size: 11px; }}
        .timestamp {{ color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏠 Phoenix Rent Growth Forecast Alert</h1>
        <p>{datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
    </div>

    <div class="alert-section">
        <h2>Alerts Summary</h2>
        <p><strong>Total Alerts:</strong> {len(alerts)}</p>
"""

        # Group alerts by severity
        critical_alerts = [a for a in alerts if a.get('severity') == 'CRITICAL']
        warning_alerts = [a for a in alerts if a.get('severity') == 'WARNING']
        info_alerts = [a for a in alerts if a.get('severity') == 'INFO']

        if critical_alerts:
            html += f"""
        <h3 style="color: #dc3545;">🚨 Critical Alerts ({len(critical_alerts)})</h3>
"""
            for alert in critical_alerts:
                timestamp = alert.get('timestamp', 'N/A')
                message = alert.get('message', 'No message')
                html += f"""
        <div class="alert alert-critical">
            <span class="severity-badge" style="background-color: #dc3545;">CRITICAL</span>
            <span class="timestamp">{timestamp}</span>
            <p>{message}</p>
        </div>
"""

        if warning_alerts:
            html += f"""
        <h3 style="color: #ffc107;">⚠️ Warnings ({len(warning_alerts)})</h3>
"""
            for alert in warning_alerts:
                timestamp = alert.get('timestamp', 'N/A')
                message = alert.get('message', 'No message')
                html += f"""
        <div class="alert alert-warning">
            <span class="severity-badge" style="background-color: #ffc107;">WARNING</span>
            <span class="timestamp">{timestamp}</span>
            <p>{message}</p>
        </div>
"""

        if info_alerts:
            html += f"""
        <h3 style="color: #17a2b8;">ℹ️ Information ({len(info_alerts)})</h3>
"""
            for alert in info_alerts:
                timestamp = alert.get('timestamp', 'N/A')
                message = alert.get('message', 'No message')
                html += f"""
        <div class="alert alert-info">
            <span class="severity-badge" style="background-color: #17a2b8;">INFO</span>
            <span class="timestamp">{timestamp}</span>
            <p>{message}</p>
        </div>
"""

        html += """
    </div>
"""

        # Add forecast summary if provided
        if forecast_summary:
            html += f"""
    <div class="summary">
        <h2>📊 Forecast Summary</h2>
        <pre style="white-space: pre-wrap; font-family: monospace;">{forecast_summary}</pre>
    </div>
"""

        html += """
    <div class="footer">
        <p>This is an automated alert from the Phoenix Rent Growth Forecast System</p>
        <p>To configure alerts, edit config/email_config.json</p>
    </div>
</body>
</html>
"""
        return html

    def send_forecast_alert(self, alerts, forecast_summary=None, attachments=None):
        """
        Send forecast alert email

        Args:
            alerts: List of alert dictionaries with 'severity', 'timestamp', 'message'
            forecast_summary: Optional forecast summary text
            attachments: Optional list of file paths to attach

        Returns:
            bool: True if sent successfully, False otherwise
        """
        if not self.enabled:
            print("Email alerts not enabled (check config/email_config.json)")
            return False

        if not self.should_send(alerts, success=True):
            print(f"Email threshold not met ({len(alerts)} alerts, not sending)")
            return False

        try:
            settings = self.config['email_settings']

            # Create message
            msg = MIMEMultipart('alternative')

            # Subject
            subject_prefix = settings.get('email_format', {}).get('subject_prefix', '[Phoenix Forecast]')
            critical_count = sum(1 for a in alerts if a.get('severity') == 'CRITICAL')
            warning_count = sum(1 for a in alerts if a.get('severity') == 'WARNING')

            if critical_count > 0:
                subject = f"{subject_prefix} 🚨 {critical_count} Critical Alerts"
            elif warning_count > 0:
                subject = f"{subject_prefix} ⚠️ {warning_count} Warnings"
            else:
                subject = f"{subject_prefix} ℹ️ Forecast Update"

            msg['Subject'] = subject
            msg['From'] = f"{settings.get('from_name', 'Forecast System')} <{settings['from_email']}>"
            msg['To'] = ', '.join(settings['recipients'])

            # Create HTML body
            html_body = self._format_alert_html(alerts, forecast_summary)
            msg.attach(MIMEText(html_body, 'html'))

            # Add attachments if specified
            if attachments and settings.get('email_format', {}).get('include_logs_attachment', False):
                for file_path in attachments:
                    if Path(file_path).exists():
                        with open(file_path, 'rb') as f:
                            part = MIMEApplication(f.read(), Name=Path(file_path).name)
                            part['Content-Disposition'] = f'attachment; filename="{Path(file_path).name}"'
                            msg.attach(part)

            # Send email
            if settings.get('use_tls', True):
                server = smtplib.SMTP(settings['smtp_server'], settings['smtp_port'])
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(settings['smtp_server'], settings['smtp_port'])

            server.login(settings['username'], settings['password'])
            server.send_message(msg)
            server.quit()

            print(f"✅ Alert email sent successfully to {len(settings['recipients'])} recipients")
            return True

        except Exception as e:
            print(f"❌ Failed to send alert email: {e}")
            return False

    def test_connection(self):
        """Test email configuration and connection"""
        if not self.enabled:
            print("❌ Email alerts are DISABLED in config/email_config.json")
            print("   Set 'enabled': true to activate")
            return False

        print("Testing email configuration...")

        settings = self.config.get('email_settings', {})

        # Validate required settings
        required_fields = ['smtp_server', 'smtp_port', 'from_email', 'username', 'password', 'recipients']
        missing_fields = [f for f in required_fields if not settings.get(f)]

        if missing_fields:
            print(f"❌ Missing required fields: {', '.join(missing_fields)}")
            return False

        # Test SMTP connection
        try:
            print(f"   Connecting to {settings['smtp_server']}:{settings['smtp_port']}...")

            if settings.get('use_tls', True):
                server = smtplib.SMTP(settings['smtp_server'], settings['smtp_port'], timeout=10)
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(settings['smtp_server'], settings['smtp_port'], timeout=10)

            print(f"   Authenticating as {settings['username']}...")
            server.login(settings['username'], settings['password'])

            print(f"✅ Connection successful!")
            server.quit()
            return True

        except smtplib.SMTPAuthenticationError:
            print("❌ Authentication failed - check username/password")
            print("   For Gmail: Use an App Password, not your regular password")
            return False
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

    def send_test_email(self):
        """Send a test email to verify configuration"""
        if not self.test_connection():
            return False

        print("\nSending test email...")

        test_alerts = [
            {
                'severity': 'INFO',
                'timestamp': datetime.now().isoformat(),
                'message': 'This is a test email from the Phoenix Forecast Alert system'
            },
            {
                'severity': 'WARNING',
                'timestamp': datetime.now().isoformat(),
                'message': 'This is a test warning alert'
            }
        ]

        test_summary = """
Phoenix Rent Growth Forecast - Test Email

2026-2028 Average: 3.55%
- 2026: 1.53%
- 2027: 4.34%
- 2028: 4.80%

This is a test forecast summary.
"""

        return self.send_forecast_alert(test_alerts, test_summary)

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Email Alert System for Phoenix Rent Growth Forecasting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test email configuration and connection
  python3 email_alerts.py --test

  # Send a test email
  python3 email_alerts.py --send-test

  # Check configuration status
  python3 email_alerts.py --status
        """
    )

    parser.add_argument('--test', action='store_true',
                       help='Test email configuration and SMTP connection')
    parser.add_argument('--send-test', action='store_true',
                       help='Send a test email')
    parser.add_argument('--status', action='store_true',
                       help='Check email configuration status')

    args = parser.parse_args()

    # Create alerter
    alerter = EmailAlerter()

    # Handle commands
    if args.test:
        alerter.test_connection()
    elif args.send_test:
        alerter.send_test_email()
    elif args.status:
        if alerter.is_enabled():
            print("✅ Email alerts are ENABLED")
            settings = alerter.config.get('email_settings', {})
            print(f"   SMTP Server: {settings.get('smtp_server')}")
            print(f"   From: {settings.get('from_email')}")
            print(f"   Recipients: {', '.join(settings.get('recipients', []))}")
        else:
            print("❌ Email alerts are DISABLED")
            print("   Configure config/email_config.json to enable")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
