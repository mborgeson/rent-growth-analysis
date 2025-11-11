# Email Alerts Setup Guide

**Project**: Rent Growth Analysis
**Created**: November 10, 2025
**Status**: Configuration Required

---

## Overview

The automation system includes professional email alert capabilities for notifying stakeholders about forecast updates, data quality issues, and validation warnings.

---

## Email Alert System Features

### Alert Types

1. **Critical Alerts** (🚨)
   - Data pipeline failures
   - Missing required columns
   - SARIMA explosive predictions (>10%)
   - System errors

2. **Warning Alerts** (⚠️)
   - Data quality issues (duplicates, missing values)
   - Poor component correlation
   - Weak Ridge regularization
   - Large forecast revisions

3. **Informational Alerts** (ℹ️)
   - Successful forecast generation
   - Model retraining notifications
   - Data quality checks passed

### Email Format

- **Subject Line**: Clear indication of alert level and issue
- **HTML Formatting**: Professional, mobile-responsive design
- **Content Sections**:
  - Executive summary
  - Detailed findings
  - Recommended actions
  - System status
  - Direct links to outputs

---

## Configuration

### Current Status

Email alerts are **DISABLED** by default for testing and configuration.

Configuration file: `config/email_config.json`

```json
{
  "email_settings": {
    "enabled": false,  // ← Currently disabled
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "use_tls": true,
    "sender_email": "",
    "sender_password": "",
    "send_on_warnings": true,
    "send_on_critical": true,
    "send_on_info": false
  },
  "recipients": {
    "primary": [],
    "cc": [],
    "critical_only": []
  }
}
```

---

## Setup Instructions

### Option 1: Gmail SMTP (Recommended for Testing)

#### Step 1: Enable 2-Factor Authentication
1. Go to Google Account settings
2. Navigate to Security → 2-Step Verification
3. Enable 2-Step Verification

#### Step 2: Generate App Password
1. Go to Google Account → Security → App passwords
2. Select app: "Mail"
3. Select device: "Other" (enter "Rent Growth Analysis")
4. Click "Generate"
5. Copy the 16-character password

#### Step 3: Update Configuration
Edit `config/email_config.json`:

```json
{
  "email_settings": {
    "enabled": true,  // ← Set to true
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "use_tls": true,
    "sender_email": "your-email@gmail.com",  // ← Your Gmail address
    "sender_password": "abcd efgh ijkl mnop",  // ← App password (16 chars)
    "send_on_warnings": true,
    "send_on_critical": true,
    "send_on_info": false
  },
  "recipients": {
    "primary": [
      "analyst@company.com"
    ],
    "cc": [
      "manager@company.com"
    ],
    "critical_only": [
      "director@company.com"
    ]
  }
}
```

### Option 2: Office 365 / Outlook SMTP

Configuration for Office 365:

```json
{
  "email_settings": {
    "enabled": true,
    "smtp_server": "smtp.office365.com",
    "smtp_port": 587,
    "use_tls": true,
    "sender_email": "your-email@company.com",
    "sender_password": "your-password",
    "send_on_warnings": true,
    "send_on_critical": true,
    "send_on_info": false
  }
}
```

### Option 3: Custom SMTP Server

For custom SMTP servers:

```json
{
  "email_settings": {
    "enabled": true,
    "smtp_server": "mail.your-domain.com",
    "smtp_port": 465,  // or 587 for TLS
    "use_tls": true,  // or false for SSL on port 465
    "sender_email": "forecasts@your-domain.com",
    "sender_password": "your-smtp-password",
    "send_on_warnings": true,
    "send_on_critical": true,
    "send_on_info": false
  }
}
```

---

## Testing Email Configuration

### Test Command

```bash
cd "/home/mattb/Rent Growth Analysis"
python3 scripts/email_alerts.py --test
```

### Expected Output

**Successful Test**:
```
✅ Email configuration test successful
✅ Test email sent to: analyst@company.com
```

**Failed Test**:
```
❌ Email configuration test failed
Error: [specific error message]
```

### Common Issues

#### "Authentication Failed"
- **Cause**: Incorrect password or 2FA not enabled (Gmail)
- **Solution**: Regenerate app password, ensure 2FA enabled

#### "Connection Refused"
- **Cause**: Wrong SMTP server or port
- **Solution**: Verify SMTP settings with your email provider

#### "TLS/SSL Error"
- **Cause**: Incorrect TLS/SSL configuration
- **Solution**: Try port 465 with TLS=false for SSL, or port 587 with TLS=true

---

## Recipient Configuration

### Recipient Types

1. **Primary Recipients**
   - Receive all enabled alert types
   - Main stakeholders who need regular updates

2. **CC Recipients**
   - Copied on all alerts
   - Secondary stakeholders for awareness

3. **Critical-Only Recipients**
   - Receive only critical alerts
   - Executives who need urgent notifications only

### Example Configuration

```json
"recipients": {
  "primary": [
    "senior.analyst@company.com",
    "data.team@company.com"
  ],
  "cc": [
    "team.lead@company.com",
    "product.manager@company.com"
  ],
  "critical_only": [
    "director@company.com",
    "vp.analytics@company.com"
  ]
}
```

---

## Alert Behavior Configuration

### Send Frequency

Control when alerts are sent:

```json
{
  "send_on_warnings": true,   // Send for warning-level issues
  "send_on_critical": true,   // Send for critical issues
  "send_on_info": false       // Send for informational updates
}
```

**Recommendations**:
- **Development/Testing**: Set all to `false`, use manual testing
- **Production**: `warnings: true`, `critical: true`, `info: false`
- **Verbose Monitoring**: Set all to `true` for comprehensive updates

---

## Integration with Automation System

### Automatic Alerts

When email alerts are enabled, the system automatically sends emails for:

1. **Monthly Forecast Runs**
   - Summary of forecast results
   - Data quality check results
   - Model validation status
   - Links to output files

2. **Data Quality Issues**
   - Missing values detected
   - Duplicate dates found
   - Extreme values identified

3. **Model Validation Failures**
   - SARIMA stability issues
   - Component correlation problems
   - Ridge regularization warnings

### Manual Alerts

Send custom alerts:

```bash
python3 scripts/email_alerts.py \
  --subject "Custom Alert Subject" \
  --message "Alert message body" \
  --severity "WARNING"  # or CRITICAL, INFO
```

---

## Security Best Practices

### Password Security

1. **Never commit credentials to git**
   - `config/email_config.json` is in `.gitignore`
   - Verify with: `git status --ignored`

2. **Use app passwords, not account passwords**
   - Gmail: Generate app-specific passwords
   - Office 365: Use modern authentication

3. **Rotate passwords regularly**
   - Change app passwords every 90 days
   - Update configuration after rotation

### Access Control

1. **Limit recipient list**
   - Only include necessary stakeholders
   - Review and update quarterly

2. **Separate production and testing**
   - Use different recipient lists for test environments
   - Add "[TEST]" prefix to test email subjects

---

## Troubleshooting

### Email Not Received

**Check**:
1. Spam/junk folder
2. Email server logs
3. Recipient email addresses correct
4. Email configuration `enabled: true`

**Debug Command**:
```bash
python3 scripts/email_alerts.py --test --verbose
```

### SSL/TLS Errors

**Solutions**:
- Port 587 → TLS enabled
- Port 465 → TLS disabled (uses implicit SSL)
- Port 25 → Usually blocked, avoid

### Authentication Issues

**Gmail**:
- Enable 2-Factor Authentication
- Generate new app password
- Use app password, not account password

**Office 365**:
- Check modern authentication enabled
- Verify account has SMTP send permissions

---

## Monitoring Email Delivery

### Log Files

Email delivery logged in:
```
logs/forecast_update_YYYYMMDD_HHMMSS.log
```

Search for email events:
```bash
grep -i "email" logs/forecast_update_*.log
```

### Delivery Confirmation

System logs show:
```
✅ Email sent successfully to 3 recipients
   - Primary: analyst@company.com, team@company.com
   - CC: manager@company.com
```

---

## Maintenance

### Weekly
- Review spam folder for false positives
- Check email delivery logs

### Monthly
- Verify recipient list current
- Test email delivery with `--test` command

### Quarterly
- Rotate app passwords
- Review and update alert thresholds
- Audit recipient access levels

---

## Contact & Support

For email configuration assistance:
1. Verify SMTP settings with email provider
2. Check email provider documentation
3. Test with `python3 scripts/email_alerts.py --test --verbose`

---

**Status**: Configuration Required
**Next Action**: Complete SMTP configuration and test
**Documentation**: Up to date as of 2025-11-10
