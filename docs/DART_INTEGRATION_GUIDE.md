# Dart Integration Guide
**Project**: Rent Growth Analysis
**Created**: November 10, 2025
**Status**: Manual sync required (Dart MCP not configured)

---

## Overview

This guide explains how to synchronize local documentation with Dart for the Rent Growth Analysis project.

---

## Dart Configuration

### Workspace Structure
- **Workspace**: `Rent Growth Analysis`
- **Dartboard**: `Rent Growth Analysis/Tasks`
- **Docs Folder**: `Rent Growth Analysis/Docs`

---

## Documentation Files to Sync

### Priority 1: Automation System Documentation

1. **AUTOMATION_SYSTEM_OVERVIEW.md** ✅ Created locally
   - Location: `docs/AUTOMATION_SYSTEM_OVERVIEW.md`
   - Dart path: `Rent Growth Analysis/Docs/AUTOMATION_SYSTEM_OVERVIEW.md`
   - Content: System architecture, components, test results, usage

2. **AUTOMATED_MONTHLY_UPDATE_GUIDE.md** ✅ Exists
   - Location: `reports/AUTOMATED_MONTHLY_UPDATE_GUIDE.md`
   - Dart path: `Rent Growth Analysis/Docs/AUTOMATED_MONTHLY_UPDATE_GUIDE.md`
   - Content: Detailed usage guide, workflow examples, troubleshooting

3. **VISUALIZATION_GUIDE.md** ✅ Exists
   - Location: `docs/VISUALIZATION_GUIDE.md`
   - Dart path: `Rent Growth Analysis/Docs/VISUALIZATION_GUIDE.md`
   - Content: Visualization module documentation, configuration, usage

### Priority 2: Analysis Documentation

4. **COMPREHENSIVE_SYSTEM_ENHANCEMENTS_ANALYSIS_20251110.txt**
   - Location: Root directory
   - Dart path: `Rent Growth Analysis/Docs/SYSTEM_ENHANCEMENTS_ANALYSIS.md`
   - Content: Complete system enhancement analysis (convert to .md)

5. **COMPREHENSIVE_ANALYSIS_USAGE_GUIDE.txt**
   - Location: `reports/`
   - Dart path: `Rent Growth Analysis/Docs/ROOT_CAUSE_ANALYSIS_GUIDE.md`
   - Content: Root cause investigation guide (convert to .md)

---

## Manual Sync Process

Since Dart MCP tool is not configured, use manual process:

### Step 1: Prepare Files

```bash
cd "/home/mattb/Rent Growth Analysis"

# Ensure all docs are in markdown format
ls -lah docs/*.md
ls -lah reports/*.md
```

### Step 2: Copy to Dart via Web Interface

1. Navigate to Dart workspace: `Rent Growth Analysis`
2. Open `Docs` folder
3. For each file:
   - Click "New Document"
   - Copy content from local file
   - Paste into Dart editor
   - Save with matching filename

### Step 3: Verify Sync

- [ ] AUTOMATION_SYSTEM_OVERVIEW.md synced
- [ ] AUTOMATED_MONTHLY_UPDATE_GUIDE.md synced
- [ ] VISUALIZATION_GUIDE.md synced
- [ ] SYSTEM_ENHANCEMENTS_ANALYSIS.md synced
- [ ] ROOT_CAUSE_ANALYSIS_GUIDE.md synced

---

## Dart MCP Tool Setup (Future)

To enable automated Dart synchronization:

### Requirements
1. Dart MCP server installation
2. Configuration in `.mcp.json`
3. Authentication credentials

### Configuration Template

```json
{
  "mcpServers": {
    "dart": {
      "command": "npx",
      "args": ["@dart/mcp-server"],
      "env": {
        "DART_API_KEY": "your-api-key",
        "DART_WORKSPACE": "Rent Growth Analysis"
      }
    }
  }
}
```

### Expected MCP Functions

Once configured, these functions should be available:
- `mcp__dart__create_document(workspace, folder, title, content)`
- `mcp__dart__update_document(workspace, folder, title, content)`
- `mcp__dart__list_documents(workspace, folder)`
- `mcp__dart__create_task(dartboard, title, description, priority, status)`
- `mcp__dart__update_task(dartboard, task_id, updates)`

---

## Task Management in Dart

### Current Tasks to Create/Update

1. **Task: Automation System Activation** ✅ COMPLETED
   - Status: Done
   - Priority: High
   - Description: Complete activation of automated monthly forecast system
   - Completion date: 2025-11-10
   - Details:
     - ✅ Configuration files created (4 files)
     - ✅ System tested successfully
     - ✅ Documentation created
     - ⏳ Cron job setup (in progress)

2. **Task: Email Alert Configuration** ⏳ PENDING
   - Status: To-do
   - Priority: Medium
   - Description: Configure SMTP settings for email alerts
   - Details:
     - Email config file created but disabled
     - Need SMTP credentials
     - Test email system

3. **Task: Data Quality Improvements** ⏳ PENDING
   - Status: To-do
   - Priority: Medium
   - Description: Fix data quality issues identified in testing
   - Details:
     - Fix duplicate date 2025-12-31
     - Add missing columns if needed by pipeline
     - Review missing value patterns

4. **Task: Cron Job Setup** 🔄 IN PROGRESS
   - Status: Doing
   - Priority: High
   - Description: Set up automated monthly execution
   - Details:
     - Setup script exists: `scripts/setup_cron.sh`
     - Need to run and verify
     - Test execution

---

## Documentation Maintenance

### When to Update Dart Docs

**Trigger Events**:
- System configuration changes
- New features added
- Major bug fixes
- Process improvements
- Quarterly reviews

### Update Process

1. Update local markdown file
2. Commit to git
3. Sync to Dart (manual or via MCP when available)
4. Add comment in Dart with update reason

---

## Best Practices

1. **Maintain Sync**: Keep local files as source of truth
2. **Version Control**: Commit all docs to git before Dart sync
3. **Clear Naming**: Use descriptive titles matching local filenames
4. **Regular Updates**: Sync docs monthly or when significant changes occur
5. **Task Alignment**: Update Dart tasks to reflect actual work status

---

## Contact

For Dart MCP tool setup assistance:
1. Check Dart MCP server documentation
2. Verify authentication credentials
3. Test MCP connection: `mcp__dart__list_documents()`

---

**Status**: Manual sync required
**Next Action**: Copy documentation files to Dart web interface
**Long-term Goal**: Configure Dart MCP tool for automated sync
