from fastapi import FastAPI, Form, UploadFile, File, Depends, HTTPException
from fastapi.security.api_key import APIKeyHeader
from crewai import Agent, Task, Crew, Process
import pandas as pd
import io
import uvicorn
from tools_qa import *;

app = FastAPI()

# --- SECURITY ---
API_TOKEN = "MY_SECRET_AGENT_KEY_123"
api_key_header = APIKeyHeader(name="access_token", auto_error=False)


@app.post("/run-qa-flow")
async def run_qa_flow(
        reqs: str = Form(...),
        url: str = Form(...),
        username: str = Form(...),
        password: str = Form(...),
        language: str = Form(...),
        framework: str = Form(...),
        locator_file: UploadFile = File(None),
        token: str = Depends(api_key_header)
):

    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized Access")

    # 1. Excel/CSV Processing
    locators_data = []
    if locator_file:
        try:
            contents = await locator_file.read()
            df = pd.read_csv(io.BytesIO(contents)) if locator_file.filename.endswith('.csv') else pd.read_excel(
                io.BytesIO(contents))
            locators_data = df.fillna("").to_dict(orient='records')
            print(f"✅ Locators Loaded: {len(locators_data)} entries.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"File error: {str(e)}")

    try:
        # 2. Agent Definition
        sdet = Agent(
            role=f'Senior {language} Automation Architect',
            goal=f'Create a complete {framework} automation suite for {url}',
            backstory=f"""You are a world-class expert in {language} and {framework}.
            Your specialty is creating professional, clean, and maintainable test code.
            You strictly follow the naming conventions of {framework}.
            You use the provided locator data to ensure tests never fail due to wrong selectors.""",
            tools=[selenium_script_writer, browser_executor, bug_analyzer_tool],
            verbose=True,
            allow_delegation=False
        )

        # 3. Task Definition (Framework Specific Logic)
        t1 = Task(
            description=f"""
            MISSION: 
            Automate the following requirement: {reqs}

            TECHNICAL STACK:
            - Language: {language}
            - Framework: {framework}

            STRICT RULES:
            1. Use ONLY these locators: {locators_data}
            2. For {framework}, follow this structure:
               - If Pytest: Create 'test_qa.py' using pytest fixtures.
               - If Behave/Cucumber: Create a 'features/' folder with .feature file and a 'steps/' folder for step definitions.
               - If TestNG: Create a Java class with @Test annotations.
            3. Use URL: {url} and Login: {username} / {password}
            4. Browser MUST be VISIBLE (Headed mode). Add 'time.sleep(5)' at the end of the script so I can see the result.

            ACTION:
            - Generate the code.
            - Use 'browser_executor' tool to run it.
            - If it fails, use 'bug_analyzer_tool' to find out why.
            """,
            expected_output=f"A professional QA report summarizing the {framework} execution results.",
            agent=sdet
        )

        # 4. Crew Execution
        crew = Crew(agents=[sdet], tasks=[t1], process=Process.sequential)
        result = crew.kickoff()

        return {"status": "Success", "report": str(result), "stack_used": f"{language}-{framework}"}

    except Exception as e:
        print(f"❌ CrewAI Error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent Failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)