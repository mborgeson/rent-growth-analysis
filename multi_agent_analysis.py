#!/usr/bin/env python3
"""
Multi-Agent Analysis System for Multifamily Rent Growth
Runs multiple analysis agents in parallel and builds consensus on relationships
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from agents.analysis_agents import (
    EconometricAgent, MachineLearningAgent, StatisticalAgent,
    ConsensusBuilder, RelationshipFinding
)
from demo_analysis import generate_synthetic_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """
    Orchestrates multiple analysis agents running in parallel
    """
    
    def __init__(self):
        self.agents = {
            'econometric': EconometricAgent(),
            'machine_learning': MachineLearningAgent(),
            'statistical': StatisticalAgent()
        }
        self.consensus_builder = ConsensusBuilder()
        self.results = {}
        
    def run_parallel_analysis(self, data: pd.DataFrame, target: str) -> Dict[str, Any]:
        """
        Run all agents in parallel and collect results
        
        Args:
            data: DataFrame with variables to analyze
            target: Target variable name
            
        Returns:
            Dictionary with all agent results and consensus
        """
        
        logger.info("="*60)
        logger.info("STARTING MULTI-AGENT PARALLEL ANALYSIS")
        logger.info("="*60)
        
        # Store agent findings
        all_findings = []
        agent_timings = {}
        
        # Run agents in parallel
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            # Submit all agent tasks
            future_to_agent = {}
            
            for agent_name, agent in self.agents.items():
                logger.info(f"🚀 Launching {agent_name} agent: {agent.name}")
                future = executor.submit(self._run_agent, agent, data, target)
                future_to_agent[future] = (agent_name, agent)
            
            # Collect results as they complete
            for future in as_completed(future_to_agent):
                agent_name, agent = future_to_agent[future]
                start_time = time.time()
                
                try:
                    findings = future.result(timeout=60)
                    elapsed = time.time() - start_time
                    
                    logger.info(f"✅ {agent_name} agent completed in {elapsed:.2f}s")
                    logger.info(f"   Found {len(findings)} variable relationships")
                    
                    all_findings.append(findings)
                    agent_timings[agent_name] = elapsed
                    
                    # Store individual agent results
                    self.results[agent_name] = {
                        'agent_id': agent.agent_id,
                        'methodology': agent.methodology,
                        'findings': findings,
                        'execution_time': elapsed
                    }
                    
                except Exception as e:
                    logger.error(f"❌ {agent_name} agent failed: {e}")
                    all_findings.append([])  # Empty findings for failed agent
        
        # Build consensus
        logger.info("\n" + "="*60)
        logger.info("BUILDING CONSENSUS FROM AGENT FINDINGS")
        logger.info("="*60)
        
        consensus = self.consensus_builder.build_consensus(all_findings)
        
        # Add timing information
        consensus['execution_metrics'] = {
            'total_time': sum(agent_timings.values()),
            'agent_timings': agent_timings,
            'parallel_speedup': sum(agent_timings.values()) / max(agent_timings.values()) if agent_timings else 1
        }
        
        self.results['consensus'] = consensus
        
        return self.results
    
    def _run_agent(self, agent, data: pd.DataFrame, target: str) -> List[RelationshipFinding]:
        """Run a single agent's analysis"""
        return agent.analyze_relationships(data, target)
    
    def print_consensus_report(self, results: Dict[str, Any]):
        """Print formatted consensus report"""
        
        consensus = results.get('consensus', {})
        summary = consensus.get('summary', {})
        
        logger.info("\n" + "="*60)
        logger.info("CONSENSUS ANALYSIS REPORT")
        logger.info("="*60)
        
        # Print methodology
        logger.info("\n📊 METHODOLOGY:")
        logger.info(consensus.get('methodology', ''))
        
        # Print summary statistics
        logger.info("\n📈 ANALYSIS SUMMARY:")
        logger.info(f"Total variables analyzed: {summary.get('total_variables_analyzed', 0)}")
        logger.info(f"Variables with predictive power: {summary.get('predictive_variables_count', 0)}")
        
        # Print confidence metrics
        conf_metrics = consensus.get('confidence_metrics', {})
        logger.info("\n🎯 CONFIDENCE METRICS:")
        logger.info(f"Mean confidence: {conf_metrics.get('mean_confidence', 0):.2%}")
        logger.info(f"Unanimous agreement rate: {conf_metrics.get('unanimous_agreement_rate', 0):.2%}")
        logger.info(f"High agreement rate: {conf_metrics.get('high_agreement_rate', 0):.2%}")
        
        # Print primary predictors
        logger.info("\n🌟 PRIMARY PREDICTORS (Strong Relationships):")
        primary = summary.get('primary_predictors', [])
        if primary:
            for pred in primary:
                var_name = pred.replace('_to_rent_growth', '')
                consensus_data = consensus['consensus_results'].get(pred, {})
                conf = consensus_data.get('consensus_confidence', 0)
                agreement = consensus_data.get('agreement_level', 'unknown')
                logger.info(f"  • {var_name}: {conf:.2%} confidence, {agreement} agreement")
                
                # Print agent-specific findings
                for agent_id, finding in consensus_data.get('agent_findings', {}).items():
                    logger.info(f"    - {agent_id}: {finding['type']} relationship ({finding['confidence']:.2%})")
        else:
            logger.info("  None identified")
        
        # Print secondary predictors
        logger.info("\n📊 SECONDARY PREDICTORS (Moderate Relationships):")
        secondary = summary.get('secondary_predictors', [])
        if secondary:
            for pred in secondary:
                var_name = pred.replace('_to_rent_growth', '')
                consensus_data = consensus['consensus_results'].get(pred, {})
                conf = consensus_data.get('consensus_confidence', 0)
                logger.info(f"  • {var_name}: {conf:.2%} confidence")
        else:
            logger.info("  None identified")
        
        # Print weak predictors
        logger.info("\n📉 WEAK PREDICTORS:")
        weak = summary.get('weak_predictors', [])
        if weak:
            for pred in weak:
                var_name = pred.replace('_to_rent_growth', '')
                logger.info(f"  • {var_name}")
        else:
            logger.info("  None identified")
        
        # Print independent variables
        logger.info("\n❌ INDEPENDENT VARIABLES (No Relationship):")
        independent = summary.get('independent_variables', [])
        if independent:
            for pred in independent:
                var_name = pred.replace('_to_rent_growth', '')
                logger.info(f"  • {var_name}")
        else:
            logger.info("  None identified")
        
        # Print execution metrics
        exec_metrics = consensus.get('execution_metrics', {})
        logger.info("\n⚡ EXECUTION METRICS:")
        logger.info(f"Total analysis time: {exec_metrics.get('total_time', 0):.2f}s")
        logger.info(f"Parallel speedup: {exec_metrics.get('parallel_speedup', 1):.1f}x")
        
        for agent, timing in exec_metrics.get('agent_timings', {}).items():
            logger.info(f"  • {agent}: {timing:.2f}s")
    
    def generate_detailed_report(self, results: Dict[str, Any], output_dir: Path):
        """Generate detailed JSON report with all findings"""
        
        # Convert findings to serializable format
        serializable_results = {}
        
        for key, value in results.items():
            if key == 'consensus':
                serializable_results[key] = value
            else:
                # Agent results
                agent_data = {
                    'agent_id': value['agent_id'],
                    'methodology': value['methodology'],
                    'execution_time': value['execution_time'],
                    'findings': []
                }
                
                for finding in value['findings']:
                    agent_data['findings'].append({
                        'variable_pair': finding.variable_pair,
                        'relationship_type': finding.relationship_type,
                        'confidence': finding.confidence,
                        'lag_structure': finding.lag_structure,
                        'direction': finding.direction,
                        'interpretation': finding.interpretation,
                        'evidence': finding.evidence
                    })
                
                serializable_results[key] = agent_data
        
        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = output_dir / f"multi_agent_consensus_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"\n📁 Detailed report saved to: {report_file}")
        
        return report_file


def main():
    """
    Main execution function
    """
    
    logger.info("="*60)
    logger.info("MULTIFAMILY RENT GROWTH - MULTI-AGENT ANALYSIS SYSTEM")
    logger.info("="*60)
    
    # Generate or load data
    logger.info("\n📊 Preparing data for analysis...")
    data = generate_synthetic_data()
    
    # Select variables for relationship analysis
    analysis_variables = [
        'rent_growth',  # Target
        'fed_funds', 'treasury_10y', 'unemployment', 'gdp_growth',
        'vacancy_rate', 'new_supply', 'absorption',
        'cpi_change', 'supply_demand_balance', 'real_rates',
        'housing_starts', 'sp500_returns'
    ]
    
    analysis_data = data[analysis_variables].copy()
    
    logger.info(f"Data shape: {analysis_data.shape}")
    logger.info(f"Variables: {list(analysis_data.columns)}")
    
    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator()
    
    # Run parallel multi-agent analysis
    results = orchestrator.run_parallel_analysis(analysis_data, 'rent_growth')
    
    # Print consensus report
    orchestrator.print_consensus_report(results)
    
    # Generate detailed report
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    report_file = orchestrator.generate_detailed_report(results, output_dir)
    
    # Print key insights
    logger.info("\n" + "="*60)
    logger.info("KEY INSIGHTS & RECOMMENDATIONS")
    logger.info("="*60)
    
    consensus = results.get('consensus', {})
    consensus_results = consensus.get('consensus_results', {})
    
    # Find strongest consensus relationships
    strong_consensus = []
    for var_pair, data in consensus_results.items():
        if (data['consensus_relationship'] == 'strong' and 
            data['agreement_level'] in ['unanimous', 'high']):
            strong_consensus.append((var_pair, data))
    
    if strong_consensus:
        logger.info("\n🎯 HIGHEST CONFIDENCE RELATIONSHIPS:")
        for var_pair, data in strong_consensus:
            var_name = var_pair.replace('_to_rent_growth', '')
            logger.info(f"\n{var_name.upper()}:")
            logger.info(f"  Recommendation: {data['recommendation']}")
            
            # Show interpretations from each agent
            logger.info("  Agent Interpretations:")
            for interp in data['interpretations'][:1]:  # Show first interpretation
                logger.info(f"    {interp}")
    
    # Summary conclusion
    logger.info("\n" + "="*60)
    logger.info("FINAL CONCLUSION")
    logger.info("="*60)
    
    summary = consensus.get('summary', {})
    primary_count = len(summary.get('primary_predictors', []))
    secondary_count = len(summary.get('secondary_predictors', []))
    
    logger.info(f"""
The multi-agent analysis has identified {primary_count} variables with strong 
predictive relationships and {secondary_count} with moderate relationships to 
multifamily rent growth.

The parallel analysis approach, combining econometric, machine learning, and 
statistical methodologies, provides robust evidence for these relationships 
through triangulation across different analytical frameworks.

Variables showing unanimous or high agreement across agents should be 
prioritized in predictive models, while those with moderate agreement warrant 
further investigation to understand the source of methodological disagreement.

This consensus-based approach reduces the risk of spurious findings and 
increases confidence in the identified relationships.
""")
    
    logger.info("="*60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    main()