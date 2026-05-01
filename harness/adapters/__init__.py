"""Adapter registry — maps config class names to adapter classes."""

from harness.adapters.ncbi_tools_adapter import NCBIToolsAdapter
from harness.adapters.geneagent_adapter import GeneAgentAdapter
from harness.adapters.genegpt_adapter import GeneGPTAdapter
from harness.adapters.genotex_adapter import GenoTEXAdapter
from harness.adapters.genomas_adapter import GenoMASAdapter
from harness.adapters.openbio_adapter import OpenBioAdapter
from harness.adapters.ehragent_adapter import EHRAgentAdapter
from harness.adapters.colacare_adapter import ColaCareAdapter
from harness.adapters.medagentbench_adapter import MedAgentBenchAdapter
from harness.adapters.mdagents_adapter import MDAgentsAdapter
from harness.adapters.txagent_adapter import TxAgentAdapter
from harness.adapters.agentclinic_adapter import AgentClinicAdapter
from harness.adapters.torchxrv_adapter import TorchXRVAdapter
from harness.adapters.medagent_pro_adapter import MedAgentProAdapter
from harness.adapters.drugagent_adapter import DrugAgentAdapter
from harness.adapters.prompt2pill_adapter import Prompt2PillAdapter
from harness.adapters.wearable_adapter import WearableAdapter
from harness.adapters.calculator_adapter import CalculatorAdapter
from harness.adapters.phenoage_adapter import PhenoAgeAdapter
from harness.adapters.chemistry_adapter import ChemistryAdapter
from harness.adapters.pytdc_adapter import PyTDCAdapter
from harness.adapters.dicom_adapter import DicomAdapter
from harness.adapters.singlecell_adapter import SingleCellAdapter
from harness.adapters.protein_adapter import ProteinAdapter
from harness.adapters.biomcp_adapter import BioMCPAdapter
from harness.adapters.tooluniverse_adapter import ToolUniverseAdapter
from harness.adapters.dicom_mcp_adapter import DicomMCPAdapter
from harness.adapters.fhir_adapter import FHIRAdapter
from harness.adapters.biomni_adapter import BiomniAdapter
from harness.adapters.txagent_stub_adapter import TxAgentStubAdapter
from harness.adapters.mcp_extra_adapters import (
    PubMedMCPAdapter, GEOMCPAdapter, UniProtMCPAdapter,
)

ADAPTER_REGISTRY: dict[str, type] = {
    "GeneAgentAdapter": GeneAgentAdapter,
    "GeneGPTAdapter": GeneGPTAdapter,
    "GenoTEXAdapter": GenoTEXAdapter,
    "GenoMASAdapter": GenoMASAdapter,
    "OpenBioAdapter": OpenBioAdapter,
    "NCBIToolsAdapter": NCBIToolsAdapter,
    "EHRAgentAdapter": EHRAgentAdapter,
    "ColaCareAdapter": ColaCareAdapter,
    "MedAgentBenchAdapter": MedAgentBenchAdapter,
    "MDAgentsAdapter": MDAgentsAdapter,
    "TxAgentAdapter": TxAgentAdapter,
    "AgentClinicAdapter": AgentClinicAdapter,
    "TorchXRVAdapter": TorchXRVAdapter,
    "MedAgentProAdapter": MedAgentProAdapter,
    "DrugAgentAdapter": DrugAgentAdapter,
    "Prompt2PillAdapter": Prompt2PillAdapter,
    "WearableAdapter": WearableAdapter,
    "CalculatorAdapter": CalculatorAdapter,
    "PhenoAgeAdapter": PhenoAgeAdapter,
    "ChemistryAdapter": ChemistryAdapter,
    "PyTDCAdapter": PyTDCAdapter,
    "DicomAdapter": DicomAdapter,
    "SingleCellAdapter": SingleCellAdapter,
    "ProteinAdapter": ProteinAdapter,
    "BioMCPAdapter": BioMCPAdapter,
    "ToolUniverseAdapter": ToolUniverseAdapter,
    "DicomMCPAdapter": DicomMCPAdapter,
    "FHIRAdapter": FHIRAdapter,
    "BiomniAdapter": BiomniAdapter,
    "TxAgentStubAdapter": TxAgentStubAdapter,
    "PubMedMCPAdapter": PubMedMCPAdapter,
    "GEOMCPAdapter": GEOMCPAdapter,
    "UniProtMCPAdapter": UniProtMCPAdapter,
}

__all__ = ["ADAPTER_REGISTRY"]
