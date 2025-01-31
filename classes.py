from typing import Optional
from pydantic import BaseModel, Field

class ClassifyQuestion(BaseModel):
    request_type: str = Field(description="Classified type of request using 'salary_request', 'vacancy_request', or 'global_question'")
    employee_code: Optional[str] = Field(None, description="The employee code extracted from the input.")

class FinalResponse(BaseModel):
    answer: str = Field(description="Final processed answer, transformed to uppercase.")

class GlobalResponse(BaseModel):
    answer: str = Field(description="Answer generated for global questions.")

class SalaryResponse(BaseModel):
    answer: str = Field(description="Fixed response for salary-related questions.")

class VacancyResponse(BaseModel):
    answer: str = Field(description="Fixed response for vacancy balance questions.")
