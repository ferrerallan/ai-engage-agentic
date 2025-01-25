from fastapi import FastAPI
from pydantic import BaseModel

# Define the input model
class EmployeeRequest(BaseModel):
    employeeCode: str

# Define the output model
class EmployeeResponse(BaseModel):
    vacancyBalanceDays: int
    YTDPayroll: float

# Initialize the FastAPI app
app = FastAPI()

# Route that receives employeeCode and returns mock data
@app.post("/employee/data", response_model=EmployeeResponse)
async def get_employee_data(request: EmployeeRequest):
    # Mock data
    if request.employeeCode == "abc123":
        return EmployeeResponse(
            vacancyBalanceDays=10,  # Mock vacation balance days
            YTDPayroll=100.00        # Mock year-to-date payroll
        )
    elif request.employeeCode == "def456":
        return EmployeeResponse(
            vacancyBalanceDays=20,  # Mock vacation balance days
            YTDPayroll=200.00    # Mock year-to-date payroll
        )
    return EmployeeResponse(
        vacancyBalanceDays=30,  # Mock vacation balance days
        YTDPayroll=300.00    # Mock year-to-date payroll
    )
