#!/bin/bash
################################################################################
# Cron Job Setup Script for Automated Monthly Forecast Updates
################################################################################
#
# Purpose: Simplify cron job installation for automated monthly forecast updates
# Usage:
#   ./setup_cron.sh install    - Install monthly cron job (1st of month, 8am)
#   ./setup_cron.sh install-weekly - Install weekly cron job (Mondays, 9am)
#   ./setup_cron.sh uninstall  - Remove cron job
#   ./setup_cron.sh status     - Check if cron job is installed
#
################################################################################

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Python interpreter
PYTHON_CMD="python3"

# Script to run
FORECAST_SCRIPT="$PROJECT_DIR/scripts/automated_monthly_forecast_update.py"

# Cron job identifier (used to find/remove the job)
CRON_IDENTIFIER="# Phoenix Rent Growth Automated Forecast Update"

# Function to display usage
show_usage() {
    echo "Usage: $0 {install|install-weekly|uninstall|status}"
    echo ""
    echo "Commands:"
    echo "  install         Install monthly cron job (1st of month at 8:00 AM)"
    echo "  install-weekly  Install weekly cron job (Mondays at 9:00 AM)"
    echo "  uninstall       Remove cron job"
    echo "  status          Check if cron job is installed"
    echo ""
    echo "Example:"
    echo "  $0 install"
    echo "  $0 status"
}

# Function to check if cron job exists
check_cron_exists() {
    crontab -l 2>/dev/null | grep -q "$CRON_IDENTIFIER"
    return $?
}

# Function to install monthly cron job
install_monthly() {
    echo -e "${YELLOW}Installing monthly cron job...${NC}"

    # Check if already installed
    if check_cron_exists; then
        echo -e "${RED}Error: Cron job already exists!${NC}"
        echo "Use '$0 uninstall' first to remove it, then reinstall."
        exit 1
    fi

    # Verify script exists
    if [ ! -f "$FORECAST_SCRIPT" ]; then
        echo -e "${RED}Error: Forecast script not found at:${NC}"
        echo "  $FORECAST_SCRIPT"
        exit 1
    fi

    # Create cron job entry
    # Runs on 1st of each month at 8:00 AM
    CRON_ENTRY="$CRON_IDENTIFIER
0 8 1 * * cd \"$PROJECT_DIR\" && $PYTHON_CMD \"$FORECAST_SCRIPT\" >> \"$PROJECT_DIR/logs/cron_execution.log\" 2>&1"

    # Add to crontab
    (crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Monthly cron job installed successfully!${NC}"
        echo ""
        echo "Schedule: 1st of each month at 8:00 AM"
        echo "Script: $FORECAST_SCRIPT"
        echo "Log: $PROJECT_DIR/logs/cron_execution.log"
        echo ""
        echo "To view current cron jobs: crontab -l"
        echo "To remove this job: $0 uninstall"
    else
        echo -e "${RED}Error: Failed to install cron job${NC}"
        exit 1
    fi
}

# Function to install weekly cron job
install_weekly() {
    echo -e "${YELLOW}Installing weekly cron job...${NC}"

    # Check if already installed
    if check_cron_exists; then
        echo -e "${RED}Error: Cron job already exists!${NC}"
        echo "Use '$0 uninstall' first to remove it, then reinstall."
        exit 1
    fi

    # Verify script exists
    if [ ! -f "$FORECAST_SCRIPT" ]; then
        echo -e "${RED}Error: Forecast script not found at:${NC}"
        echo "  $FORECAST_SCRIPT"
        exit 1
    fi

    # Create cron job entry
    # Runs every Monday at 9:00 AM
    CRON_ENTRY="$CRON_IDENTIFIER
0 9 * * 1 cd \"$PROJECT_DIR\" && $PYTHON_CMD \"$FORECAST_SCRIPT\" >> \"$PROJECT_DIR/logs/cron_execution.log\" 2>&1"

    # Add to crontab
    (crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Weekly cron job installed successfully!${NC}"
        echo ""
        echo "Schedule: Every Monday at 9:00 AM"
        echo "Script: $FORECAST_SCRIPT"
        echo "Log: $PROJECT_DIR/logs/cron_execution.log"
        echo ""
        echo "To view current cron jobs: crontab -l"
        echo "To remove this job: $0 uninstall"
    else
        echo -e "${RED}Error: Failed to install cron job${NC}"
        exit 1
    fi
}

# Function to uninstall cron job
uninstall() {
    echo -e "${YELLOW}Uninstalling cron job...${NC}"

    # Check if cron job exists
    if ! check_cron_exists; then
        echo -e "${RED}Error: Cron job not found${NC}"
        echo "Nothing to uninstall."
        exit 1
    fi

    # Remove cron job (remove the identifier line and the line after it)
    crontab -l 2>/dev/null | grep -v "$CRON_IDENTIFIER" | grep -v "cd \"$PROJECT_DIR\" && $PYTHON_CMD" | crontab -

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Cron job uninstalled successfully!${NC}"
    else
        echo -e "${RED}Error: Failed to uninstall cron job${NC}"
        exit 1
    fi
}

# Function to check status
check_status() {
    echo -e "${YELLOW}Checking cron job status...${NC}"
    echo ""

    if check_cron_exists; then
        echo -e "${GREEN}✅ Cron job is INSTALLED${NC}"
        echo ""
        echo "Current cron configuration:"
        echo "---"
        crontab -l 2>/dev/null | grep -A 1 "$CRON_IDENTIFIER"
        echo "---"
    else
        echo -e "${RED}❌ Cron job is NOT installed${NC}"
        echo ""
        echo "To install: $0 install"
    fi
}

# Main script logic
case "$1" in
    install)
        install_monthly
        ;;
    install-weekly)
        install_weekly
        ;;
    uninstall)
        uninstall
        ;;
    status)
        check_status
        ;;
    *)
        show_usage
        exit 1
        ;;
esac

exit 0
