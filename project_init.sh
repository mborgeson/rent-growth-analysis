#!/bin/bash

################################################################################
# Multifamily Rent Growth Analysis - Project Initialization Script
# Version: 1.0.0
# Date: 2025
# Purpose: Comprehensive project setup with agent orchestration initialization
################################################################################

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script configuration
PROJECT_NAME="multifamily-rent-growth-analysis"
PYTHON_VERSION="3.10"
NODE_VERSION="18"
LOG_FILE="initialization.log"
ERROR_LOG="initialization_errors.log"

# Function: Print colored messages
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ${GREEN}✓${NC} $1" | tee -a "$LOG_FILE"
}

print_warning() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ${YELLOW}⚠${NC} $1" | tee -a "$LOG_FILE"
}

print_error() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ${RED}✗${NC} $1" | tee -a "$ERROR_LOG"
}

print_section() {
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" | tee -a "$LOG_FILE"
    echo -e "${MAGENTA}$1${NC}" | tee -a "$LOG_FILE"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n" | tee -a "$LOG_FILE"
}

# Function: Check prerequisites
check_prerequisites() {
    print_section "CHECKING PREREQUISITES"
    
    local missing_prereqs=()
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        python_version=$(python3 --version | cut -d' ' -f2)
        # Use Python itself to check version instead of bc
        if python3 -c "import sys; exit(0 if sys.version_info >= (3,9) else 1)" 2>/dev/null; then
            print_status "Python $python_version detected"
        else
            missing_prereqs+=("Python >= 3.9")
        fi
    else
        missing_prereqs+=("Python 3")
    fi
    
    # Check Node.js
    if command -v node &> /dev/null; then
        node_version=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
        if [[ $node_version -ge 16 ]]; then
            print_status "Node.js v$(node --version) detected"
        else
            missing_prereqs+=("Node.js >= 16")
        fi
    else
        missing_prereqs+=("Node.js")
    fi
    
    # Check Git
    if command -v git &> /dev/null; then
        print_status "Git $(git --version | cut -d' ' -f3) detected"
    else
        missing_prereqs+=("Git")
    fi
    
    # Check Docker (optional but recommended)
    if command -v docker &> /dev/null; then
        print_status "Docker detected (optional)"
    else
        print_warning "Docker not found (optional but recommended for database containers)"
    fi
    
    # Report missing prerequisites
    if [[ ${#missing_prereqs[@]} -gt 0 ]]; then
        print_error "Missing prerequisites: ${missing_prereqs[*]}"
        echo "Please install missing prerequisites and rerun the script."
        exit 1
    fi
    
    print_status "All required prerequisites met"
}

# Function: Create directory structure
create_directory_structure() {
    print_section "CREATING PROJECT DIRECTORY STRUCTURE"
    
    # Main project directories
    directories=(
        "data/raw/national"
        "data/raw/msa/phoenix"
        "data/raw/msa/austin"
        "data/raw/msa/dallas"
        "data/raw/msa/denver"
        "data/raw/msa/salt_lake_city"
        "data/raw/msa/nashville"
        "data/raw/msa/miami"
        "data/processed/national"
        "data/processed/msa"
        "data/interim"
        "data/external"
        "data/cache"
        
        "src/data_acquisition"
        "src/data_processing"
        "src/feature_engineering"
        "src/models/econometric"
        "src/models/machine_learning"
        "src/models/ensemble"
        "src/analysis"
        "src/validation"
        "src/visualization"
        "src/utils"
        "src/api"
        
        "agents/data_swarm"
        "agents/analysis_swarm"
        "agents/validation_swarm"
        "agents/visualization_swarm"
        "agents/orchestrator"
        "agents/consensus"
        "agents/monitoring"
        
        "config/api_keys"
        "config/databases"
        "config/models"
        "config/agents"
        
        "notebooks/exploratory"
        "notebooks/analysis"
        "notebooks/modeling"
        "notebooks/validation"
        
        "tests/unit"
        "tests/integration"
        "tests/e2e"
        "tests/fixtures"
        
        "docs/api"
        "docs/methodology"
        "docs/results"
        "docs/deployment"
        
        "outputs/reports"
        "outputs/figures"
        "outputs/models"
        "outputs/predictions"
        
        "logs/agents"
        "logs/models"
        "logs/data"
        "logs/errors"
        
        "deployment/docker"
        "deployment/kubernetes"
        "deployment/scripts"
        
        "monitoring/dashboards"
        "monitoring/alerts"
        "monitoring/metrics"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_status "Created directory: $dir"
    done
    
    # Create .gitkeep files to preserve empty directories
    find . -type d -empty -exec touch {}/.gitkeep \;
}

# Function: Initialize Git repository
initialize_git() {
    print_section "INITIALIZING GIT REPOSITORY"
    
    if [[ ! -d .git ]]; then
        git init
        print_status "Git repository initialized"
    else
        print_warning "Git repository already exists"
    fi
    
    # Create comprehensive .gitignore
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv
*.egg-info/
dist/
build/
*.egg

# Jupyter Notebooks
.ipynb_checkpoints
*/.ipynb_checkpoints/*

# Data files
data/raw/*
data/processed/*
data/interim/*
data/cache/*
!data/**/.gitkeep
*.csv
*.parquet
*.feather
*.h5
*.hdf5
*.pkl
*.pickle

# API Keys and Secrets
config/api_keys/*
!config/api_keys/.gitkeep
.env
.env.local
.env.*.local
*.key
*.pem
*.crt

# Logs
logs/**/*.log
logs/**/*.txt
!logs/**/.gitkeep

# Model outputs
outputs/models/*.pkl
outputs/models/*.joblib
outputs/models/*.h5
outputs/predictions/*.csv
!outputs/**/.gitkeep

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Testing
.coverage
htmlcov/
.pytest_cache/
.tox/

# Documentation
docs/_build/
docs/.doctrees/

# Node.js
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Docker
*.log
docker-compose.override.yml

# Monitoring
monitoring/metrics/*.db
monitoring/dashboards/*.json
!monitoring/**/.gitkeep
EOF
    
    print_status "Created comprehensive .gitignore file"
}

# Function: Create Python virtual environment
setup_python_environment() {
    print_section "SETTING UP PYTHON ENVIRONMENT"
    
    # Create virtual environment
    python3 -m venv venv
    print_status "Created Python virtual environment"
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    print_status "Upgraded pip, setuptools, and wheel"
    
    # Create requirements files
    cat > requirements.txt << 'EOF'
# Data manipulation and analysis
# python>=3.9
pandas>=2.1.0
numpy>=1.26.0
scipy>=1.11.0
scikit-learn>=1.3.2
polars>=0.20.0

# Time series analysis
statsmodels>=0.14.1
arch>=6.0.0
pmdarima>=2.0.0
prophet>=1.1.0
tslearn>=0.6.0
sktime>=0.20.0

# Econometric models
linearmodels>=5.0
pandas-datareader>=0.10.0

# Machine learning
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
tensorflow>=2.13.0
torch>=2.0.0
pytorch-lightning>=2.0.0

# Deep learning for time series
darts>=0.25.0
neuralprophet>=0.5.0

# API clients
fredapi>=0.5.0
alpha-vantage>=2.3.0
yfinance>=0.2.0
quandl>=3.7.0

# Database connectivity
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
pymongo>=4.0.0
redis>=5.0.0
clickhouse-driver>=0.2.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
bokeh>=3.0.0
altair>=5.0.0
holoviews>=1.17.0

# Parallel processing
dask[complete]>=2023.0.0
ray>=2.0.0
joblib>=1.3.0
multiprocess>=0.70.0

# Testing and validation
pytest>=7.4.0
pytest-cov>=4.0.0
pytest-asyncio>=0.21.0
hypothesis>=6.80.0
great-expectations>=0.17.0

# Code quality
black>=23.0.0
flake8>=6.0.0
mypy>=1.4.0
pylint>=2.17.0
pre-commit>=3.3.0

# Documentation
sphinx>=7.0.0
sphinx-rtd-theme>=1.3.0
mkdocs>=1.5.0
mkdocs-material>=9.0.0

# Monitoring and logging
prometheus-client>=0.17.0
structlog>=23.0.0
loguru>=0.7.0
sentry-sdk>=1.30.0

# Agent orchestration
asyncio>=3.4.3
aiohttp>=3.8.0
aiofiles>=23.0.0
aiokafka>=0.8.0
celery>=5.3.0
flower>=2.0.0
dramatiq>=1.14.0

# Web framework (for API)
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
httpx>=0.24.0

# Utilities
python-dotenv>=1.0.0
click>=8.1.0
rich>=13.5.0
tqdm>=4.66.0
tenacity>=8.2.0
cachetools>=5.3.0
EOF
    
    print_status "Created requirements.txt with comprehensive dependencies"
    
    # Create development requirements
    cat > requirements-dev.txt << 'EOF'
# Development tools
ipython>=8.14.0
jupyter>=1.0.0
jupyterlab>=4.0.0
notebook>=7.0.0

# Debugging
ipdb>=0.13.0
pudb>=2022.1.0
snoop>=0.4.0

# Profiling
memory-profiler>=0.61.0
line-profiler>=4.0.0
py-spy>=0.3.0
scalene>=1.5.0

# Additional testing tools
pytest-benchmark>=4.0.0
pytest-mock>=3.11.0
pytest-xdist>=3.3.0
locust>=2.15.0

# Documentation tools
pydoc-markdown>=4.8.0
autodoc>=0.5.0
EOF
    
    print_status "Created requirements-dev.txt"
}

# Function: Install Python dependencies
install_python_dependencies() {
    print_section "INSTALLING PYTHON DEPENDENCIES"
    
    source venv/bin/activate
    
    # Install main requirements
    pip install -r requirements.txt
    print_status "Installed main Python dependencies"
    
    # Install development requirements
    pip install -r requirements-dev.txt
    print_status "Installed development Python dependencies"
    
    # Install Jupyter kernel
    python -m ipykernel install --user --name "$PROJECT_NAME" --display-name "Rent Growth Analysis"
    print_status "Installed Jupyter kernel"
}

# Function: Setup Node.js dependencies for agent orchestration
setup_nodejs_environment() {
    print_section "SETTING UP NODE.JS ENVIRONMENT"
    
    # Initialize package.json
    cat > package.json << 'EOF'
{
  "name": "multifamily-rent-growth-agents",
  "version": "1.0.0",
  "description": "Agent orchestration system for multifamily rent growth analysis",
  "main": "agents/orchestrator/index.js",
  "scripts": {
    "start": "node agents/orchestrator/index.js",
    "dev": "nodemon agents/orchestrator/index.js",
    "test": "jest",
    "lint": "eslint agents/**/*.js",
    "monitor": "node agents/monitoring/dashboard.js"
  },
  "dependencies": {
    "axios": "^1.5.0",
    "bull": "^4.11.0",
    "bullmq": "^4.12.0",
    "express": "^4.18.0",
    "socket.io": "^4.6.0",
    "redis": "^4.6.0",
    "winston": "^3.11.0",
    "dotenv": "^16.3.0",
    "uuid": "^9.0.0",
    "lodash": "^4.17.21",
    "moment": "^2.29.0",
    "node-cron": "^3.0.0",
    "p-queue": "^7.4.0",
    "prom-client": "^15.0.0"
  },
  "devDependencies": {
    "nodemon": "^3.0.0",
    "jest": "^29.7.0",
    "eslint": "^8.50.0",
    "prettier": "^3.0.0"
  }
}
EOF
    
    print_status "Created package.json for agent orchestration"
    
    # Install Node.js dependencies
    npm install
    print_status "Installed Node.js dependencies"
}

# Function: Create environment configuration
create_environment_config() {
    print_section "CREATING ENVIRONMENT CONFIGURATION"
    
    # Create .env template
    cat > .env.template << 'EOF'
# API Configuration
FRED_API_KEY=your_fred_api_key_here
CENSUS_API_KEY=your_census_api_key_here
BLS_API_KEY=your_bls_api_key_here
QUANDL_API_KEY=your_quandl_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/rent_growth
REDIS_URL=redis://localhost:6379
MONGODB_URL=mongodb://localhost:27017/rent_growth
CLICKHOUSE_URL=clickhouse://localhost:8123

# Agent Configuration
AGENT_ORCHESTRATOR_PORT=8080
AGENT_MONITORING_PORT=8081
AGENT_CONSENSUS_THRESHOLD=0.75
AGENT_MAX_WORKERS=10
AGENT_TIMEOUT_SECONDS=300

# Model Configuration
MODEL_CACHE_DIR=./outputs/models
MODEL_VERSION_CONTROL=true
MODEL_AUTO_RETRAIN=true
MODEL_PERFORMANCE_THRESHOLD=0.85

# Monitoring Configuration
PROMETHEUS_ENDPOINT=http://localhost:9090
GRAFANA_ENDPOINT=http://localhost:3000
SENTRY_DSN=your_sentry_dsn_here
LOG_LEVEL=INFO

# Computational Resources
MAX_MEMORY_GB=32
MAX_CPU_CORES=16
GPU_ENABLED=false
PARALLEL_WORKERS=8

# Data Configuration
DATA_UPDATE_FREQUENCY=monthly
DATA_VALIDATION_ENABLED=true
DATA_QUALITY_THRESHOLD=0.95
CACHE_EXPIRY_HOURS=24

# Security
API_RATE_LIMIT=100
API_RATE_WINDOW_MINUTES=15
ENABLE_SSL=true
JWT_SECRET=your_jwt_secret_here
EOF
    
    print_status "Created .env.template file"
    
    # Copy template to .env if it doesn't exist
    if [[ ! -f .env ]]; then
        cp .env.template .env
        print_warning "Created .env file from template - please update with actual values"
    else
        print_warning ".env file already exists - skipping"
    fi
}

# Function: Initialize agent orchestration system
initialize_agent_orchestration() {
    print_section "INITIALIZING AGENT ORCHESTRATION SYSTEM"
    
    # Copy agent orchestration config
    if [[ -f "agent_orchestration_config.yaml" ]]; then
        cp agent_orchestration_config.yaml config/agents/
        print_status "Copied agent orchestration configuration"
    else
        print_warning "agent_orchestration_config.yaml not found - skipping"
    fi
    
    # Create orchestrator initialization script
    cat > agents/orchestrator/init.py << 'EOF'
#!/usr/bin/env python3
"""
Agent Orchestrator Initialization
Validates and initializes the multi-agent system
"""

import yaml
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OrchestratorInitializer:
    """Initialize and validate agent orchestration system"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.validation_results = {}
    
    def load_config(self) -> Dict[str, Any]:
        """Load orchestration configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)
    
    def validate_swarm_configuration(self) -> bool:
        """Validate swarm agent configurations"""
        logger.info("Validating swarm configurations...")
        
        required_swarms = ['data_swarm', 'analysis_swarm', 'validation_swarm', 'visualization_swarm']
        swarms = self.config.get('swarms', {})
        
        for swarm_name in required_swarms:
            if swarm_name not in swarms:
                logger.error(f"Missing required swarm: {swarm_name}")
                return False
            
            swarm = swarms[swarm_name]
            
            # Validate agent count
            if len(swarm.get('agents', [])) < 2:
                logger.warning(f"Swarm {swarm_name} has less than 2 agents")
            
            # Validate agent capabilities
            for agent in swarm.get('agents', []):
                if not agent.get('capabilities'):
                    logger.error(f"Agent {agent.get('name')} has no capabilities defined")
                    return False
        
        logger.info("✓ Swarm configuration validated successfully")
        return True
    
    def validate_consensus_mechanisms(self) -> bool:
        """Validate consensus mechanisms"""
        logger.info("Validating consensus mechanisms...")
        
        consensus = self.config.get('consensus', {})
        
        # Check voting threshold
        threshold = consensus.get('voting_threshold', 0)
        if threshold < 0.5 or threshold > 1.0:
            logger.error(f"Invalid voting threshold: {threshold}")
            return False
        
        # Check quorum requirements
        quorum = consensus.get('quorum_percentage', 0)
        if quorum < 0.5:
            logger.warning(f"Low quorum percentage: {quorum}")
        
        logger.info("✓ Consensus mechanisms validated successfully")
        return True
    
    def validate_execution_flow(self) -> bool:
        """Validate execution flow configuration"""
        logger.info("Validating execution flow...")
        
        flow = self.config.get('execution_flow', {})
        phases = flow.get('phases', [])
        
        required_phases = ['initialization', 'data_acquisition', 'analysis', 'validation', 'reporting']
        
        for required_phase in required_phases:
            if not any(phase.get('name') == required_phase for phase in phases):
                logger.error(f"Missing required phase: {required_phase}")
                return False
        
        # Check phase dependencies
        for phase in phases:
            deps = phase.get('depends_on', [])
            for dep in deps:
                if not any(p.get('name') == dep for p in phases):
                    logger.error(f"Invalid dependency {dep} in phase {phase.get('name')}")
                    return False
        
        logger.info("✓ Execution flow validated successfully")
        return True
    
    def initialize_agent_directories(self) -> bool:
        """Create necessary agent directories and files"""
        logger.info("Initializing agent directories...")
        
        try:
            # Create agent module files
            swarms = self.config.get('swarms', {})
            
            for swarm_name, swarm_config in swarms.items():
                swarm_dir = Path(f"agents/{swarm_name}")
                swarm_dir.mkdir(parents=True, exist_ok=True)
                
                # Create __init__.py
                init_file = swarm_dir / "__init__.py"
                if not init_file.exists():
                    init_file.write_text(f'"""Agent swarm: {swarm_name}"""')
                
                # Create agent files
                for agent in swarm_config.get('agents', []):
                    agent_file = swarm_dir / f"{agent['name'].lower().replace(' ', '_')}.py"
                    if not agent_file.exists():
                        agent_file.write_text(f'''"""
Agent: {agent['name']}
Capabilities: {', '.join(agent.get('capabilities', []))}
"""

class {agent['name'].replace(' ', '')}:
    """Implementation for {agent['name']}"""
    
    def __init__(self):
        self.name = "{agent['name']}"
        self.capabilities = {agent.get('capabilities', [])}
    
    async def execute(self, task):
        """Execute assigned task"""
        # TODO: Implement agent logic
        pass
''')
                
                logger.info(f"✓ Initialized {swarm_name} directory structure")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize agent directories: {e}")
            return False
    
    async def test_connectivity(self) -> bool:
        """Test connectivity to required services"""
        logger.info("Testing service connectivity...")
        
        # Test Redis connection
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0, socket_connect_timeout=5)
            r.ping()
            logger.info("✓ Redis connection successful")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
        
        # Test database connection
        try:
            import psycopg2
            from urllib.parse import urlparse
            import os
            
            db_url = os.getenv('DATABASE_URL', '')
            if db_url:
                result = urlparse(db_url)
                conn = psycopg2.connect(
                    database=result.path[1:],
                    user=result.username,
                    password=result.password,
                    host=result.hostname,
                    port=result.port,
                    connect_timeout=5
                )
                conn.close()
                logger.info("✓ PostgreSQL connection successful")
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed: {e}")
        
        return True
    
    async def run_initialization(self) -> bool:
        """Run complete initialization process"""
        logger.info("="*60)
        logger.info("Starting Agent Orchestration System Initialization")
        logger.info("="*60)
        
        # Run validation checks
        checks = [
            ("Swarm Configuration", self.validate_swarm_configuration),
            ("Consensus Mechanisms", self.validate_consensus_mechanisms),
            ("Execution Flow", self.validate_execution_flow),
            ("Agent Directories", self.initialize_agent_directories),
            ("Service Connectivity", self.test_connectivity)
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            logger.info(f"\nRunning: {check_name}")
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            self.validation_results[check_name] = result
            if not result:
                all_passed = False
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("INITIALIZATION SUMMARY")
        logger.info("="*60)
        
        for check, result in self.validation_results.items():
            status = "✓ PASSED" if result else "✗ FAILED"
            logger.info(f"{check:.<40} {status}")
        
        if all_passed:
            logger.info("\n✅ Agent Orchestration System initialized successfully!")
        else:
            logger.error("\n❌ Initialization failed - please fix issues and retry")
        
        return all_passed

async def main():
    """Main initialization function"""
    config_path = "config/agents/agent_orchestration_config.yaml"
    
    if not Path(config_path).exists():
        logger.error(f"Configuration file not found: {config_path}")
        logger.info("Please ensure agent_orchestration_config.yaml is in place")
        sys.exit(1)
    
    initializer = OrchestratorInitializer(config_path)
    success = await initializer.run_initialization()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())
EOF
    
    print_status "Created orchestrator initialization script"
    
    # Make the script executable
    chmod +x agents/orchestrator/init.py
}

# Function: Create Docker configuration
create_docker_config() {
    print_section "CREATING DOCKER CONFIGURATION"
    
    # Create docker-compose.yml
    cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    container_name: rent_growth_postgres
    environment:
      POSTGRES_DB: rent_growth
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: secure_password_here
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - rent_growth_network

  redis:
    image: redis:7-alpine
    container_name: rent_growth_redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - rent_growth_network

  mongodb:
    image: mongo:6
    container_name: rent_growth_mongodb
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: secure_password_here
      MONGO_INITDB_DATABASE: rent_growth
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
    networks:
      - rent_growth_network

  clickhouse:
    image: clickhouse/clickhouse-server:latest
    container_name: rent_growth_clickhouse
    ports:
      - "8123:8123"
      - "9000:9000"
    volumes:
      - clickhouse_data:/var/lib/clickhouse
    networks:
      - rent_growth_network

  prometheus:
    image: prom/prometheus:latest
    container_name: rent_growth_prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    networks:
      - rent_growth_network

  grafana:
    image: grafana/grafana:latest
    container_name: rent_growth_grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/dashboards:/etc/grafana/provisioning/dashboards
    networks:
      - rent_growth_network

  jupyter:
    build:
      context: .
      dockerfile: deployment/docker/Dockerfile.jupyter
    container_name: rent_growth_jupyter
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/home/jovyan/work
      - ./data:/home/jovyan/data
      - ./src:/home/jovyan/src
    environment:
      JUPYTER_ENABLE_LAB: "yes"
    networks:
      - rent_growth_network

networks:
  rent_growth_network:
    driver: bridge

volumes:
  postgres_data:
  redis_data:
  mongodb_data:
  clickhouse_data:
  prometheus_data:
  grafana_data:
EOF
    
    print_status "Created docker-compose.yml"
    
    # Create Dockerfile for Jupyter
    mkdir -p deployment/docker
    cat > deployment/docker/Dockerfile.jupyter << 'EOF'
FROM jupyter/scipy-notebook:latest

USER root

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

USER $NB_UID

# Copy requirements and install Python packages
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install Jupyter extensions
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager \
    @jupyterlab/git \
    @jupyterlab/debugger

WORKDIR /home/jovyan/work
EOF
    
    print_status "Created Jupyter Dockerfile"
}

# Function: Create initial test suite
create_test_suite() {
    print_section "CREATING INITIAL TEST SUITE"
    
    # Create pytest configuration
    cat > pytest.ini << 'EOF'
[tool:pytest]
minversion = 7.0
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -ra
    --strict-markers
    --cov=src
    --cov-report=term-missing
    --cov-report=html
    --cov-report=xml
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow tests
    api: API tests
    model: Model tests
    agent: Agent tests
EOF
    
    print_status "Created pytest configuration"
    
    # Create sample test file
    cat > tests/unit/test_initialization.py << 'EOF'
"""
Initial test suite to verify project setup
"""

import pytest
import sys
from pathlib import Path

def test_python_version():
    """Test Python version requirement"""
    assert sys.version_info >= (3, 9), "Python 3.9+ required"

def test_project_structure():
    """Test that required directories exist"""
    required_dirs = [
        "data", "src", "agents", "config", 
        "notebooks", "tests", "docs", "outputs"
    ]
    
    for dir_name in required_dirs:
        assert Path(dir_name).exists(), f"Directory {dir_name} not found"

def test_configuration_files():
    """Test that configuration files exist"""
    required_files = [
        "requirements.txt",
        ".env.template",
        "pytest.ini"
    ]
    
    for file_name in required_files:
        assert Path(file_name).exists(), f"File {file_name} not found"

def test_agent_orchestration_config():
    """Test agent orchestration configuration exists"""
    config_path = Path("config/agents/agent_orchestration_config.yaml")
    if config_path.exists():
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert 'orchestrator' in config
        assert 'swarms' in config
        assert 'consensus' in config
        assert 'execution_flow' in config

@pytest.mark.integration
def test_database_connectivity():
    """Test database connectivity (requires services running)"""
    import os
    
    # This test requires DATABASE_URL to be set
    if os.getenv('DATABASE_URL'):
        import psycopg2
        from urllib.parse import urlparse
        
        db_url = os.getenv('DATABASE_URL')
        result = urlparse(db_url)
        
        try:
            conn = psycopg2.connect(
                database=result.path[1:],
                user=result.username,
                password=result.password,
                host=result.hostname,
                port=result.port,
                connect_timeout=5
            )
            conn.close()
            assert True
        except Exception as e:
            pytest.fail(f"Database connection failed: {e}")
EOF
    
    print_status "Created initial test suite"
}

# Function: Create Makefile for common tasks
create_makefile() {
    print_section "CREATING MAKEFILE"
    
    cat > Makefile << 'EOF'
.PHONY: help init test clean run-agents run-jupyter run-services stop-services

help:
	@echo "Available commands:"
	@echo "  make init          - Initialize project environment"
	@echo "  make test          - Run test suite"
	@echo "  make clean         - Clean temporary files"
	@echo "  make run-agents    - Start agent orchestration system"
	@echo "  make run-jupyter   - Start Jupyter Lab"
	@echo "  make run-services  - Start Docker services"
	@echo "  make stop-services - Stop Docker services"

init:
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	npm install

test:
	. venv/bin/activate && pytest

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

run-agents:
	. venv/bin/activate && python agents/orchestrator/init.py
	npm run start

run-jupyter:
	. venv/bin/activate && jupyter lab --port=8888

run-services:
	docker-compose up -d

stop-services:
	docker-compose down

validate-agents:
	. venv/bin/activate && python agents/orchestrator/init.py

update-deps:
	. venv/bin/activate && pip install --upgrade -r requirements.txt
	npm update

lint:
	. venv/bin/activate && black src/ agents/ tests/
	. venv/bin/activate && flake8 src/ agents/ tests/
	. venv/bin/activate && mypy src/ agents/

generate-docs:
	. venv/bin/activate && sphinx-build -b html docs docs/_build
EOF
    
    print_status "Created Makefile"
}

# Function: Generate README
generate_readme() {
    print_section "GENERATING README"
    
    cat > README.md << 'EOF'
# Multifamily Rent Growth Time Series Analysis

## Overview
Comprehensive time series analysis system for multifamily rent growth prediction using econometric models, machine learning, and multi-agent orchestration.

## Project Structure
```
.
├── data/                 # Data storage (raw, processed, interim)
├── src/                  # Source code
├── agents/              # Multi-agent orchestration system
├── config/              # Configuration files
├── notebooks/           # Jupyter notebooks
├── tests/               # Test suites
├── docs/                # Documentation
├── outputs/             # Model outputs and reports
├── logs/                # Application logs
├── deployment/          # Deployment configurations
└── monitoring/          # Monitoring and metrics
```

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+
- Docker (optional but recommended)
- Git

### Installation
```bash
# Run initialization script
./project_init.sh

# Or manually:
make init
```

### Configuration
1. Copy `.env.template` to `.env`
2. Update API keys and database credentials
3. Configure agent orchestration parameters

### Running the System

#### Start Services
```bash
make run-services  # Start Docker containers
```

#### Initialize Agents
```bash
make validate-agents  # Validate agent configuration
make run-agents      # Start agent orchestration
```

#### Launch Jupyter
```bash
make run-jupyter     # Start Jupyter Lab
```

### Testing
```bash
make test           # Run test suite
make lint           # Run code quality checks
```

## Agent Orchestration

The system uses a hierarchical multi-agent architecture:

- **Orchestrator**: Central coordination and task distribution
- **Data Swarm**: Data acquisition and preprocessing
- **Analysis Swarm**: Statistical and ML model execution
- **Validation Swarm**: Model validation and backtesting
- **Visualization Swarm**: Results presentation and reporting

## Data Sources

### National Level
- FRED (Federal Reserve Economic Data)
- Census Bureau
- Bureau of Labor Statistics
- Financial markets data

### MSA Level
- Phoenix, Austin, Dallas, Denver
- Salt Lake City, Nashville, Miami
- Local economic indicators
- Regional real estate metrics

## Models

### Econometric
- VAR/VECM
- ARDL
- Granger Causality
- Cointegration Analysis

### Machine Learning
- Random Forest
- XGBoost/LightGBM
- LSTM Networks
- Ensemble Methods

## Documentation
See `/docs` directory for detailed documentation:
- API Documentation
- Methodology Guide
- Deployment Instructions
- Agent Orchestration Manual

## License
Proprietary - All Rights Reserved

## Contact
For questions or support, contact the development team.
EOF
    
    print_status "Generated README.md"
}

# Function: Final validation
final_validation() {
    print_section "RUNNING FINAL VALIDATION"
    
    source venv/bin/activate
    
    # Run agent orchestration validation
    if [[ -f "agents/orchestrator/init.py" ]]; then
        print_status "Validating agent orchestration system..."
        python agents/orchestrator/init.py
    fi
    
    # Run initial tests
    print_status "Running initial test suite..."
    pytest tests/unit/test_initialization.py -v
    
    print_status "Final validation complete"
}

# Function: Print summary
print_summary() {
    print_section "INITIALIZATION COMPLETE"
    
    echo -e "${GREEN}✓ Project structure created${NC}"
    echo -e "${GREEN}✓ Git repository initialized${NC}"
    echo -e "${GREEN}✓ Python environment configured${NC}"
    echo -e "${GREEN}✓ Dependencies installed${NC}"
    echo -e "${GREEN}✓ Agent orchestration system initialized${NC}"
    echo -e "${GREEN}✓ Docker configuration created${NC}"
    echo -e "${GREEN}✓ Test suite established${NC}"
    echo -e "${GREEN}✓ Documentation generated${NC}"
    
    echo -e "\n${CYAN}Next Steps:${NC}"
    echo "1. Update .env file with your API keys and credentials"
    echo "2. Start Docker services: make run-services"
    echo "3. Validate agents: make validate-agents"
    echo "4. Run agents: make run-agents"
    echo "5. Launch Jupyter: make run-jupyter"
    
    echo -e "\n${YELLOW}Important Files to Review:${NC}"
    echo "- .env (configure API keys)"
    echo "- config/agents/agent_orchestration_config.yaml"
    echo "- docker-compose.yml (adjust resource limits)"
    echo "- requirements.txt (verify package versions)"
    
    echo -e "\n${GREEN}Project initialization successful!${NC}"
}

# Main execution
main() {
    echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║   Multifamily Rent Growth Analysis - Project Initialization   ║${NC}"
    echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════════╝${NC}"
    
    # Create log files
    touch "$LOG_FILE" "$ERROR_LOG"
    
    # Run initialization steps
    check_prerequisites
    create_directory_structure
    initialize_git
    setup_python_environment
    install_python_dependencies
    setup_nodejs_environment
    create_environment_config
    initialize_agent_orchestration
    create_docker_config
    create_test_suite
    create_makefile
    generate_readme
    final_validation
    print_summary
}

# Execute main function
main "$@"