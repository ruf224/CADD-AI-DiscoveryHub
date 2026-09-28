
import json
from pathlib import Path
import pickle
import numpy as np
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, Crippen, rdMolDescriptors
from rdkit.Chem.Draw import MolToImage


# ============================================================
# CADD-AI-DiscoveryHub
# ============================================================

st.set_page_config(
    page_title="CADD-AI-DiscoveryHub",
    page_icon="🧬",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "solubility_model.pkl"
METADATA_PATH = BASE_DIR / "solubility_metadata.json"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


@st.cache_data
def load_metadata():

    if METADATA_PATH.exists():

        with open(METADATA_PATH, "r") as f:
            return json.load(f)

    return {}


model = load_model()
metadata = load_metadata()


# ============================================================
# MOLECULAR FUNCTIONS
# ============================================================

def parse_smiles(smiles):

    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)

    return mol


def molecular_fingerprint(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048
    )

    return np.array(fp)


def predict_solubility(smiles):

    fp = molecular_fingerprint(smiles)

    if fp is None:
        return None

    prediction = model.predict(
        fp.reshape(1, -1)
    )[0]

    return float(prediction)


def calculate_descriptors(mol):

    return {
        "Molecular Weight": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "H-bond Donors": Lipinski.NumHDonors(mol),
        "H-bond Acceptors": Lipinski.NumHAcceptors(mol),
        "Rotatable Bonds": Lipinski.NumRotatableBonds(mol),
        "Aromatic Rings": Lipinski.NumAromaticRings(mol),
        "Heavy Atoms": Lipinski.HeavyAtomCount(mol),
        "Formal Charge": Chem.GetFormalCharge(mol)
    }


def lipinski_assessment(desc):

    violations = 0

    if desc["Molecular Weight"] > 500:
        violations += 1

    if desc["LogP"] > 5:
        violations += 1

    if desc["H-bond Donors"] > 5:
        violations += 1

    if desc["H-bond Acceptors"] > 10:
        violations += 1

    return violations


# ============================================================
# HEADER
# ============================================================

st.title("🧬 CADD-AI-DiscoveryHub")

st.markdown(
    """
    ### AI-assisted Computer-Aided Drug Discovery Platform

    **Current validated module:** Molecular property prediction

    Enter a compound as a SMILES string to explore its molecular
    properties and predicted aqueous solubility.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("CADD Workflow")

st.sidebar.markdown(
    """
    **01 — Compound Input**  
    **02 — Molecular Profile**  
    **03 — ADMET Prediction**  
    **04 — Target Prediction**  
    **05 — Virtual Screening**  
    **06 — Docking**  
    **07 — Molecular Dynamics**  
    **08 — Lead Optimization**
    """
)

st.sidebar.divider()

st.sidebar.caption(
    "Validated ML model: Delaney ESOL-derived Random Forest"
)


# ============================================================
# COMPOUND INPUT
# ============================================================

st.header("1. Compound Input")

default_smiles = "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"

smiles = st.text_input(
    "Enter SMILES",
    value=default_smiles
)


# ============================================================
# ANALYSIS
# ============================================================

if smiles:

    mol = parse_smiles(smiles)

    if mol is None:

        st.error(
            "Invalid SMILES. Please enter a valid molecular structure."
        )

    else:

        st.success("Valid molecular structure detected.")

        canonical_smiles = Chem.MolToSmiles(mol)

        # ----------------------------------------------------
        # MOLECULAR PROFILE
        # ----------------------------------------------------

        st.header("2. Molecular Profile")

        col1, col2 = st.columns([1, 2])

        with col1:

            image = MolToImage(
                mol,
                size=(350, 350)
            )

            st.image(
                image,
                caption="2D molecular structure"
            )

        with col2:

            st.write("**Canonical SMILES**")

            st.code(canonical_smiles)

            descriptors = calculate_descriptors(mol)

            descriptor_rows = []

            for key, value in descriptors.items():

                if isinstance(value, float):
                    value = round(value, 3)

                descriptor_rows.append(
                    {
                        "Property": key,
                        "Value": value
                    }
                )

            st.dataframe(
                descriptor_rows,
                use_container_width=True,
                hide_index=True
            )


        # ----------------------------------------------------
        # DRUG-LIKENESS
        # ----------------------------------------------------

        st.header("3. Drug-Likeness")

        violations = lipinski_assessment(
            descriptors
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Lipinski violations",
                violations
            )

        with col2:
            if violations <= 1:
                st.success("Generally compatible with Lipinski criteria")
            else:
                st.warning("Multiple Lipinski criteria exceeded")

        with col3:
            st.metric(
                "Molecular weight",
                f'{descriptors["Molecular Weight"]:.1f} Da'
            )


        # ----------------------------------------------------
        # SOLUBILITY
        # ----------------------------------------------------

        st.header("4. AI Solubility Prediction")

        prediction = predict_solubility(smiles)

        if prediction is not None:

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Predicted logS",
                    f"{prediction:.3f}"
                )

            with col2:

                st.info(
                    "logS represents predicted aqueous solubility "
                    "on the model's training scale."
                )

            st.subheader("Model information")

            info = {
                "Dataset": metadata.get(
                    "dataset",
                    "Delaney ESOL"
                ),
                "Algorithm": metadata.get(
                    "algorithm",
                    "Random Forest"
                ),
                "Training compounds": metadata.get(
                    "training_samples",
                    "N/A"
                ),
                "Validation R²": metadata.get(
                    "validation_R2",
                    "N/A"
                ),
                "Test R²": metadata.get(
                    "test_R2",
                    "N/A"
                )
            }

            st.dataframe(
                [
                    {"Parameter": k, "Value": v}
                    for k, v in info.items()
                ],
                use_container_width=True,
                hide_index=True
            )


        # ----------------------------------------------------
        # CADD WORKFLOW
        # ----------------------------------------------------

        st.header("5. Integrated CADD Workflow")

        tabs = st.tabs(
            [
                "ADMET",
                "Target Prediction",
                "Virtual Screening",
                "Docking",
                "Molecular Dynamics",
                "Lead Optimization"
            ]
        )

        # ----------------------------------------------------
        # ADMET
        # ----------------------------------------------------

        with tabs[0]:

            st.subheader("ADMET Assessment")

            st.info(
                "The solubility module is currently backed by "
                "a trained machine-learning model."
            )

            admet_table = [
                {
                    "Property": "Aqueous solubility",
                    "Status": "ML prediction",
                    "Result": f"logS = {prediction:.3f}"
                },
                {
                    "Property": "Molecular weight",
                    "Status": "Descriptor",
                    "Result": f'{descriptors["Molecular Weight"]:.2f} Da'
                },
                {
                    "Property": "LogP",
                    "Status": "Descriptor",
                    "Result": f'{descriptors["LogP"]:.2f}'
                },
                {
                    "Property": "TPSA",
                    "Status": "Descriptor",
                    "Result": f'{descriptors["TPSA"]:.2f} Å²'
                }
            ]

            st.dataframe(
                admet_table,
                use_container_width=True,
                hide_index=True
            )


        # ----------------------------------------------------
        # TARGET PREDICTION
        # ----------------------------------------------------

        with tabs[1]:

            st.subheader("Target Prediction")

            st.info(
                "Target prediction module — integration layer prepared."
            )

            st.markdown(
                """
                Planned workflow:

                **SMILES → molecular representation → target model →
                ranked protein targets**
                """
            )


        # ----------------------------------------------------
        # VIRTUAL SCREENING
        # ----------------------------------------------------

        with tabs[2]:

            st.subheader("Virtual Screening")

            st.info(
                "Virtual screening interface prepared."
            )

            st.markdown(
                """
                Future screening workflow:

                **Compound library → molecular fingerprints →
                similarity/filtering → prioritized candidates**
                """
            )


        # ----------------------------------------------------
        # DOCKING
        # ----------------------------------------------------

        with tabs[3]:

            st.subheader("Molecular Docking")

            st.warning(
                "Docking engine not executed in the current deployment."
            )

            st.markdown(
                """
                Intended workflow:

                **Target structure → ligand preparation →
                docking → binding pose → interaction analysis**
                """
            )

            st.caption(
                "This interface is an architectural placeholder and "
                "does not represent a completed docking calculation."
            )


        # ----------------------------------------------------
        # MD
        # ----------------------------------------------------

        with tabs[4]:

            st.subheader("Molecular Dynamics")

            st.warning(
                "Real-time molecular dynamics is not executed in this "
                "browser deployment."
            )

            st.markdown(
                """
                Intended workflow:

                **Protein–ligand complex → MD simulation →
                trajectory analysis → stability assessment**
                """
            )


        # ----------------------------------------------------
        # LEAD OPTIMIZATION
        # ----------------------------------------------------

        with tabs[5]:

            st.subheader("AI Lead Optimization")

            st.info(
                "Lead optimization module prepared for integration."
            )

            st.markdown(
                """
                Example optimization objectives:

                - improve predicted solubility
                - maintain molecular weight
                - control lipophilicity
                - maintain drug-like properties
                - reduce undesirable structural features
                """
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CADD-AI-DiscoveryHub | AI-assisted research demonstrator | "
    "Predictions should be experimentally validated."
)
