from fastapi import FastAPI, Form, UploadFile, File, Depends, HTTPException
from fastapi.security.api_key import APIKeyHeader
from crewai import Agent, Task, Crew, Process
import pandas as pd
import io
import uvicorn
from tools_qa import *

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

    locators_data = []
    if locator_file:
        try:
            contents = await locator_file.read()
            df = pd.read_csv(io.BytesIO(contents)) if locator_file.filename.endswith('.csv') else pd.read_excel(
                io.BytesIO(contents))
            locators_data = df.fillna("").to_dict(orient='records')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"File error: {str(e)}")

    try:
        # 2. Agent:
        sdet = Agent(
            role=f'Elite {language} {framework} Execution Robot',
            goal=f'Run automation visibly and generate a professional dashboard-style report.',
            backstory=f"""You are a precise QA Robot. You communicate with 'Sir'.
            Your primary mission is to TRIGGER the browser. You are FORBIDDEN from giving 
            a final answer until you have called the 'browser_executor' tool and captured its output.
            You follow clean coding standards and Page Object Model.""",
            tools=[selenium_script_writer, browser_executor, bug_analyzer_tool],
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )

        # 3. Task
        t1 = Task(
            description=f"""
            MISSION: 
            Automate these steps: {reqs}
            Use: {url}
            {username}
            {password}
            call {browser_executor}
            Create the code as per {language} 
            if {framework} == "Cucumber"
                create feature file , Stepdefinitions , Page Objects and TestRunner etc.
                create full framework of the cucumber framework and save it to Local system location.
            if {framework} == "Behave"
                Create feature file, stepdefinitions, Page Objects etc
                All code files should be in VS Code Style
                Create full framework of the behave framework and save it to Local system location
            if {framework} == "TestNG"
                Create full framework of the TestNG framework and Save it to Local System Location
                All code files should be in VS Code Style
            if {framework} == "Pytest"
                Create full framework of the Pytest framework and save it to Local system location
                All code files should be in VS Code Style
            if {framework} == "RobotFramework"
                Create full framework of the RobotFramework framework and save it to Local system location
                All code files should be in VS Code Style
            

            STRICT EXECUTION RULES:
            1. Use ONLY locators: {locators_data}.
            2. MANDATORY: You MUST call 'browser_executor' with your generated code. 
            3. NO HEADLESS: Browser must be visible (Headed).
            4. Add 'time.sleep(10)' at the very end of the script before closing.
            CRITICAL BROWSER PERSISTENCE RULES:
            1. MANDATORY: Use ChromeOptions and add: options.add_experimental_option("detach", True)
            2. Use 'driver.maximize_window()' right after opening.
            3. Add 'time.sleep(15)' at the very end (after all steps).
            4. Do NOT use 'driver.quit()' or 'driver.close()' anywhere in the script. I want to manually inspect the result.

            --- FINAL REPORT TEMPLATE (EDITOR LOOK) ---
            Your response must start with 'MISSION REPORT ACCOMPLISHED, SIR! 🤖'
            Then use this structure:
            # 📊 MISSION SUMMARY
            - **Status:** [SUCCESS 🟢 / FAILED 🔴]
            - **Stack:** {language} / {framework}

            ### 💻 SOURCE CODE (VS CODE STYLE)
            Use triple backticks with the language name for syntax highlighting.
            ```{language.lower()}
            (Full professional code here)
            ```

            ### 🚀 EXECUTION LOGS
            ```text
            (Raw output from the browser_executor tool)
            ```
            """,
            expected_output=f"A structured report that MUST include executed steps from the tool.",
            agent=sdet
        )

        crew = Crew(agents=[sdet], tasks=[t1], process=Process.sequential)
        result = crew.kickoff()

        return {"status": "Success", "report": str(result), "stack_used": f"{language}-{framework}"}

    except Exception as e:
        print(f"❌ CrewAI Error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent Failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)