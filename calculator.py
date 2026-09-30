import streamlit as st

st.title("🧮 Calculator App")

# รับค่าตัวเลข 2 จำนวน
num1 = st.number_input("Enter first number", value=0.0)
num2 = st.number_input("Enter second number", value=0.0)

# เลือกการดำเนินการ
operation = st.selectbox(
    "Select operation",
    ("+ (Addition)", "- (Subtraction)", "* (Multiplication)", "/ (Division)"),
)

# ปุ่มคำนวณผลลัพธ์
if st.button("Calculate"):
    if operation == "+ (Addition)":
        result = num1 + num2
    elif operation == "- (Subtraction)":
        result = num1 - num2
    elif operation == "* (Multiplication)":
        result = num1 * num2
    elif operation == "/ (Division)":
        if num2 != 0:
            result = num1 / num2
        else:
            result = "Error: Division by zero"

    st.success(f"Result: {result}")