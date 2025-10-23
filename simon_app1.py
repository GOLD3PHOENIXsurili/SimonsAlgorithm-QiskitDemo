# simon_final.py - COMPLETE CORRECTED VERSION
import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter

# Set page configuration
st.set_page_config(
    page_title="Simon's Algorithm Simulator",
    page_icon="⚛️",
    layout="wide"
)

# CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1a5276;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        background: linear-gradient(135deg, #1a5276, #3498db);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 2px solid #27ae60;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #155724;
    }
    .info-box {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border: 2px solid #1976d2;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #0d47a1;
    }
    .theory-box {
        background: linear-gradient(135deg, #fff3e0, #ffcc80);
        border: 2px solid #f57c00;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #e65100;
    }
    .circuit-box {
        background-color: #2c3e50;
        border: 2px solid #34495e;
        border-radius: 10px;
        padding: 20px;
        font-family: 'Courier New', monospace;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #ecf0f1;
    }
    .step-box {
        background: linear-gradient(135deg, #e8f5e8, #c8e6c9);
        border: 2px solid #4caf50;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        color: #1b5e20;
    }
    .math-box {
        background-color: #f8f9fa;
        border: 2px solid #6c757d;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        font-family: 'Cambria Math', serif;
        color: #2c3e50;
    }
    .dark-text {
        color: #2c3e50 !important;
    }
    .white-box-fix {
        background-color: #ffffff;
        border: 2px solid #bdc3c7;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

def generate_random_secret(n):
    """Generate a random non-zero binary secret string"""
    while True:
        secret = ''.join(random.choice(['0', '1']) for _ in range(n))
        if secret != '0' * n:
            return secret

def simulate_quantum_circuit(secret_string, shots=1000):
    """
    Simulate Simon's Algorithm quantum circuit
    Returns measurement outcomes that are orthogonal to the secret
    """
    n = len(secret_string)
    s_vector = [int(bit) for bit in secret_string]
    measurements = []
    
    for _ in range(shots):
        # In Simon's algorithm, we ONLY measure vectors orthogonal to s
        while True:
            y_vector = [random.choice([0, 1]) for _ in range(n)]
            dot_product = sum(s_vector[i] * y_vector[i] for i in range(n)) % 2
            # Only accept vectors that are orthogonal to s
            if dot_product == 0:
                break
                
        measurements.append(''.join(str(bit) for bit in y_vector))
    
    return Counter(measurements)

def solve_linear_system_gf2(measurements, n):
    """
    Solve the linear system over GF(2) to find the secret string s
    We need to find s such that s·y = 0 mod 2 for all measured y
    """
    # Convert measurements to binary vectors
    vectors = []
    for bitstring in measurements:
        vector = [int(bit) for bit in bitstring]
        vectors.append(vector)
    
    # Remove duplicate vectors and zero vector
    unique_vectors = []
    for vec in vectors:
        if vec not in unique_vectors and vec != [0]*n:
            unique_vectors.append(vec)
    
    st.write(f"**Unique non-zero vectors found:** {len(unique_vectors)}")
    
    if len(unique_vectors) < n - 1:
        st.warning(f"Need at least {n-1} linearly independent vectors. Got {len(unique_vectors)}.")
        return None
    
    # Try all possible non-zero secrets
    possible_secrets = []
    for candidate in range(1, 2**n):
        s_candidate = [int(bit) for bit in format(candidate, f'0{n}b')]
        
        # Check if this candidate is orthogonal to ALL measured vectors
        is_valid = True
        for vector in unique_vectors:
            dot_product = sum(s_candidate[i] * vector[i] for i in range(n)) % 2
            if dot_product != 0:
                is_valid = False
                break
        
        if is_valid:
            secret_str = ''.join(str(bit) for bit in s_candidate)
            possible_secrets.append(secret_str)
    
    return possible_secrets

def gaussian_elimination_gf2(vectors, n):
    """
    Perform Gaussian elimination over GF(2) to find the secret
    More sophisticated approach
    """
    if len(vectors) < n - 1:
        return None
    
    # Build matrix from vectors
    matrix = vectors.copy()
    
    # Perform Gaussian elimination
    for col in range(n):
        # Find pivot
        pivot_row = None
        for row in range(col, len(matrix)):
            if matrix[row][col] == 1:
                pivot_row = row
                break
        
        if pivot_row is None:
            continue
            
        # Swap rows
        matrix[col], matrix[pivot_row] = matrix[pivot_row], matrix[col]
        
        # Eliminate below and above
        for row in range(len(matrix)):
            if row != col and matrix[row][col] == 1:
                for j in range(n):
                    matrix[row][j] = (matrix[row][j] + matrix[col][j]) % 2
    
    # Find non-zero vector in null space
    for candidate in range(1, 2**n):
        s = [int(bit) for bit in format(candidate, f'0{n}b')]
        valid = True
        for row in matrix:
            if sum(row[i] * s[i] for i in range(n)) % 2 != 0:
                valid = False
                break
        if valid:
            return [''.join(str(bit) for bit in s)]
    
    return None

def demonstrate_orthogonality(secret, measurements):
    """Show that measured vectors are orthogonal to secret"""
    s_vec = [int(bit) for bit in secret]
    results = []
    
    for bitstring in list(measurements)[:5]:  # Check first 5 measurements
        v_vec = [int(bit) for bit in bitstring]
        dot_product = sum(s_vec[i] * v_vec[i] for i in range(len(secret))) % 2
        results.append((bitstring, dot_product))
    
    return results

def draw_quantum_circuit(n, secret_string):
    """Generate a visual representation of the quantum circuit"""
    circuit = f"""
    SIMON'S ALGORITHM QUANTUM CIRCUIT (n={n}, s={secret_string})
    {'=' * 70}
    
    INPUT REGISTER ({n} qubits):
    """
    
    # Input qubits
    for i in range(n):
        circuit += f"q{i} |0⟩ --[H]--|         |--[H]--| Measurement → y{i}\n"
    
    circuit += f"\nOUTPUT REGISTER ({n} qubits):\n"
    for i in range(n):
        circuit += f"q{n+i} |0⟩ --| SIMON |--| f(x) ⟩\n"
    
    circuit += f"""
    {'=' * 70}
    
    QUANTUM GATES:
    • [H] = Hadamard gate (creates superposition)
    • [SIMON] = Oracle implementing f(x) = f(x ⊕ {secret_string})
    • Measurement in computational basis
    
    CIRCUIT STEPS:
    1. Initialize |0⟩^({2*n})
    2. Apply H^⊗{n} to input register
    3. Apply Simon Oracle
    4. Apply H^⊗{n} to input register  
    5. Measure input register
    """
    
    return circuit

def main():
    # Header with better contrast
    st.markdown('<div class="main-header">⚛️ Simon\'s Algorithm Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; font-size: 1.2rem; color: #2c3e50; margin-bottom: 2rem;">Demonstrating Quantum Exponential Speedup</div>', unsafe_allow_html=True)
    
    # FIXED: Theory section with proper visibility
    st.markdown('<div class="section-header">🎯 What is Simon\'s Problem?</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="white-box-fix">
    <h3 style="color: #2c3e50;">Simon's Problem Definition</h3>
    
    Given a function <strong>f: {0,1}ⁿ → {0,1}ⁿ</strong> that satisfies 
    <strong>f(x) = f(x ⊕ s)</strong> for some unknown binary string <strong>s ≠ 0</strong>, find <strong>s</strong>.
    
    <div style="margin-top: 15px;">
    <table style="width: 100%; border-collapse: collapse;">
        <tr style="background-color: #f8f9fa;">
            <td style="padding: 10px; border: 1px solid #dee2e6;"><strong>Classical Complexity</strong></td>
            <td style="padding: 10px; border: 1px solid #dee2e6;">Θ(2<sup>n/2</sup>) queries</td>
            <td style="padding: 10px; border: 1px solid #dee2e6; color: #e74c3c;">Exponential time</td>
        </tr>
        <tr style="background-color: #f8f9fa;">
            <td style="padding: 10px; border: 1px solid #dee2e6;"><strong>Quantum Complexity</strong></td>
            <td style="padding: 10px; border: 1px solid #dee2e6;">Θ(n) queries</td>
            <td style="padding: 10px; border: 1px solid #dee2e6; color: #27ae60;">Polynomial time</td>
        </tr>
    </table>
    </div>
    
    <div style="margin-top: 15px; padding: 15px; background-color: #e8f4fd; border-radius: 5px;">
    <strong>Quantum Advantage:</strong> Exponential speedup demonstrated!
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Algorithm Steps with better visibility
    st.markdown('<div class="section-header">🔬 Algorithm Steps</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="white-box-fix">
    <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px;">
        <div style="background: linear-gradient(135deg, #e3f2fd, #bbdefb); padding: 15px; border-radius: 8px; text-align: center; border: 2px solid #1976d2;">
            <h4 style="color: #0d47a1; margin: 0;">1️⃣ Initialize</h4>
            <p style="color: #1565c0; margin: 5px 0 0 0;">|0⟩ⁿ|0⟩ⁿ</p>
        </div>
        <div style="background: linear-gradient(135deg, #e8f5e9, #c8e6c9); padding: 15px; border-radius: 8px; text-align: center; border: 2px solid #388e3c;">
            <h4 style="color: #1b5e20; margin: 0;">2️⃣ Superposition</h4>
            <p style="color: #2e7d32; margin: 5px 0 0 0;">H gates on input</p>
        </div>
        <div style="background: linear-gradient(135deg, #fff3e0, #ffcc80); padding: 15px; border-radius: 8px; text-align: center; border: 2px solid #f57c00;">
            <h4 style="color: #e65100; margin: 0;">3️⃣ Oracle</h4>
            <p style="color: #ef6c00; margin: 5px 0 0 0;">Apply f(x)</p>
        </div>
        <div style="background: linear-gradient(135deg, #fce4ec, #f8bbd9); padding: 15px; border-radius: 8px; text-align: center; border: 2px solid #c2185b;">
            <h4 style="color: #880e4f; margin: 0;">4️⃣ Interference</h4>
            <p style="color: #ad1457; margin: 5px 0 0 0;">H gates again</p>
        </div>
        <div style="background: linear-gradient(135deg, #e1f5fe, #b3e5fc); padding: 15px; border-radius: 8px; text-align: center; border: 2px solid #0288d1;">
            <h4 style="color: #01579b; margin: 0;">5️⃣ Measure</h4>
            <p style="color: #0277bd; margin: 5px 0 0 0;">Get y: s·y=0</p>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    # # Mathematical Explanation with proper formatting
    # st.markdown('<div class="section-header">🧮 Mathematical Foundation</div>', unsafe_allow_html=True)
    
    # st.markdown("""
    # <div class="white-box-fix">
    # <h3 style="color: #2c3e50;">Key Equations</h3>
    
    # <div class="math-box">
    # <strong>Simon's Condition:</strong><br>
    # $$f(x) = f(x \\oplus s) \\quad \\forall x$$
    # </div>
    
    # <div class="math-box">
    # <strong>After second Hadamard:</strong><br>
    # $$\\frac{1}{2^n} \\sum_y \\sum_x (-1)^{x \\cdot y} |y\\rangle |f(x)\\rangle$$
    # </div>
    
    # <div class="math-box">
    # <strong>Measurement probability:</strong><br>
    # $$
    # P(y) = 
    # \\begin{cases} 
    # \\frac{1}{2^{n-1}} & \\text{if } s \\cdot y = 0 \\\\
    # 0 & \\text{if } s \\cdot y = 1 
    # \\end{cases}
    # $$
    # </div>
    
    # <div style="margin-top: 15px; padding: 15px; background-color: #e3f2fd; border-radius: 5px;">
    # <strong>Key Insight:</strong> Quantum interference eliminates all states where s·y = 1, leaving only orthogonal measurements!
    # </div>
    # </div>
    # """, unsafe_allow_html=True)




    # Mathematical Explanation with proper LaTeX formatting
    st.markdown('<div class="section-header">🧮 Mathematical Foundation</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="white-box-fix">
    <h3 style="color: #2c3e50;">Key Equations</h3>
    </div>
    """, unsafe_allow_html=True)

    # Simon's Condition
    st.markdown("""
    <div class="math-box">
    <strong>Simon's Condition:</strong><br>
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"f(x) = f(x \oplus s) \quad \forall x")

    # After second Hadamard
    st.markdown("""
    <div class="math-box">
    <strong>After second Hadamard:</strong><br>
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"\frac{1}{2^n} \sum_y \sum_x (-1)^{x \cdot y} |y\rangle |f(x)\rangle")

    # Measurement probability
    st.markdown("""
    <div class="math-box">
    <strong>Measurement probability:</strong><br>
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"""
    P(y) = 
    \begin{cases} 
    \frac{1}{2^{n-1}} & \text{if } s \cdot y = 0 \\
    0 & \text{if } s \cdot y = 1 
    \end{cases}
    """)

    # Key Insight
    st.markdown("""
    <div style="margin-top: 15px; padding: 15px; background-color: #e3f2fd; border-radius: 5px;">
    <strong>Key Insight:</strong> Quantum interference eliminates all states where $s \cdot y = 1$, leaving only orthogonal measurements!
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)






    
    # Sidebar Controls
    st.sidebar.markdown('<div class="section-header">🎛️ Controls</div>', unsafe_allow_html=True)
    
    n = st.sidebar.slider("**Qubits (n)**", 2, 6, 3, 
                         help="Length of secret string")
    
    secret_option = st.sidebar.radio("**Secret**", ["Default", "Custom", "Random"])
    
    if secret_option == "Default":
        secret_string = "110" if n == 3 else generate_random_secret(n)
    elif secret_option == "Custom":
        default_secret = "110" if n == 3 else generate_random_secret(n)
        custom_secret = st.sidebar.text_input("**Secret string**", default_secret)
        if custom_secret and len(custom_secret) == n and all(c in '01' for c in custom_secret) and custom_secret != '0'*n:
            secret_string = custom_secret
        else:
            secret_string = default_secret
    else:
        if st.sidebar.button("**Random**"):
            secret_string = generate_random_secret(n)
        else:
            secret_string = "110" if n == 3 else generate_random_secret(n)
    
    st.sidebar.write(f"**Current Secret:** `{secret_string}`")
    
    shots = st.sidebar.slider("**Shots**", 500, 5000, 2000,
                            help="More shots = better chance to recover secret")
    
    # Run Simulation
    if st.button("🚀 **Run Simon's Algorithm**", type="primary", use_container_width=True):
        st.markdown("---")
        
        with st.spinner("Executing quantum circuit..."):
            # Display Circuit
            st.markdown("### 🔌 Quantum Circuit")
            circuit_diagram = draw_quantum_circuit(n, secret_string)
            st.markdown(f'<div class="circuit-box">{circuit_diagram}</div>', unsafe_allow_html=True)
            
            # Run simulation
            counts = simulate_quantum_circuit(secret_string, shots)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Measurement Results")
                st.write(f"**Total shots:** {shots}")
                st.write(f"**Unique outcomes:** {len(counts)}")
                st.write(f"**Secret to find:** `{secret_string}`")
                
                # Show top measurements
                st.write("**Most frequent measurements:**")
                for bitstring, count in list(counts.most_common(8)):
                    percentage = (count / shots) * 100
                    st.write(f"`{bitstring}`: {count} times ({percentage:.1f}%)")
            
            with col2:
                st.markdown("### 📈 Measurement Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Get top measurements for plotting
                top_bitstrings = list(counts.keys())[:10]
                top_counts = [counts[b] for b in top_bitstrings]
                
                bars = ax.bar(top_bitstrings, top_counts, color='skyblue', edgecolor='navy')
                ax.set_xlabel('Measured Bitstrings')
                ax.set_ylabel('Count')
                ax.set_title('Quantum Measurement Outcomes')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{int(height)}', ha='center', va='bottom')
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
        
        # Solve for secret
        st.markdown("---")
        st.markdown("### 🔍 Recovering the Secret")
        
        # Try both methods
        possible_secrets = solve_linear_system_gf2(counts.keys(), n)
        
        if not possible_secrets:
            st.info("Trying alternative solving method...")
            vectors_for_gaussian = [[int(bit) for bit in bs] for bs in counts.keys() if bs != '0'*n]
            possible_secrets = gaussian_elimination_gf2(vectors_for_gaussian, n)
        
        if possible_secrets:
            st.markdown(f"""
            <div class="success-box">
                <h3>✅ Quantum Algorithm Successful!</h3>
                <p><strong>Original Secret:</strong> <code>{secret_string}</code></p>
                <p><strong>Recovered Secret(s):</strong> {', '.join(f'<code>{s}</code>' for s in possible_secrets)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Verify orthogonality
            st.markdown("#### 🧮 Orthogonality Verification")
            st.write("All measured vectors should satisfy **s·y = 0 mod 2**:")
            
            verification_results = demonstrate_orthogonality(secret_string, counts.keys())
            for bitstring, dot_product in verification_results[:5]:
                status = "✅" if dot_product == 0 else "❌"
                st.write(f"{status} s·{bitstring} = {dot_product} mod 2")
                
        else:
            st.error("""
            **❌ Could not recover secret. Try:**
            - Increasing the number of shots (2000+ recommended)
            - Using a different secret string
            - The algorithm needs enough linearly independent measurements
            """)
            
            # Debug information
            with st.expander("🔧 Debug Information"):
                unique_vectors = []
                for bs in counts.keys():
                    if bs != '0'*n:
                        vec = [int(bit) for bit in bs]
                        if vec not in unique_vectors:
                            unique_vectors.append(vec)
                
                st.write(f"Unique non-zero vectors: {len(unique_vectors)}")
                st.write(f"Need at least: {n-1} for {n}-bit secret")
                st.write("Top measurements:", list(counts.most_common(10)))
    
    # Educational content
    st.markdown("---")
    st.markdown("### 📚 How It Works")
    
    st.markdown("""
    **Key Quantum Mechanics:**
    - **Superposition**: Evaluate all inputs simultaneously
    - **Interference**: Cancel out wrong answers, amplify correct ones  
    - **Measurement**: Collapse to states that reveal information about s

    **Why it works:**
    """)

    # Equation 1: First Hadamard
    st.markdown("1. First Hadamard creates:")
    st.latex(r"\frac{1}{\sqrt{2^n}} \sum_x |x\rangle|0\rangle")

    # Equation 2: Oracle
    st.markdown("2. Oracle creates:")
    st.latex(r"\frac{1}{\sqrt{2^n}} \sum_x |x\rangle|f(x)\rangle")

    st.markdown("""
    3. Second Hadamard causes interference
    4. Only states with $s \\cdot y = 0$ survive measurement

    **After n-1 measurements**, we can solve for s using linear algebra over GF(2)!
    """)

if __name__ == "__main__":
    main()

    