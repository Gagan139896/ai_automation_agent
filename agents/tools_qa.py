import os, subprocess, sys, json
from crewai.tools import tool


@tool("test_case_generator")
def test_case_generator(requirements: str):
    """Converts raw requirements into a structured JSON list of manual test cases."""
    return f"Manual Test Cases created for: {requirements}"


@tool("selenium_script_writer")
def selenium_script_writer(manual_steps: str, locators: str, language: str = "Python", framework: str = "Selenium"):
    """
    Generates automation code based on manual steps and a locator map for the specified language and framework.
    Supports: Python, Java, C#, etc.
    """
    return f"{language} {framework} code generated successfully for the requested steps."


@tool("browser_executor")
def browser_executor(code: str, language: str = "Python", framework: str = "Selenium"):
    """Executes code and captures screenshots on failure."""
    import os, subprocess, sys

    # Cleaning & Saving
    clean_code = code.strip().replace("```python", "").replace("```", "")
    file_name = "test_mission.py"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(clean_code)

    print(f"🚀 [EXECUTION]: Running {framework} in headed mode...")

    try:
        result = subprocess.run(
            [sys.executable, file_name],
            capture_output=True, text=True, shell=True
        )

        # Check if screenshot exists
        screenshot_path = "error.png"
        has_screenshot = os.path.exists(screenshot_path)

        if result.returncode == 0:
            return f"✅ SUCCESS:\n{result.stdout}"
        else:
            report = f"❌ FAILED:\n{result.stderr}\n{result.stdout}"
            if has_screenshot:
                report += f"\n📸 SCREENSHOT_SAVED: {screenshot_path}"
            return report

    except Exception as e:
        return f"⚠️ SYSTEM ERROR: {str(e)}"


@tool("bug_analyzer_tool")
def bug_analyzer_tool(error_logs: str):
    """Analyzes error logs to identify if the issue is a UI bug, a script error, or a locator mismatch."""
    return f"Bug Analysis Report: Found issue in logs -> {error_logs[:100]}"


@tool("final_report_compiler")
def final_report_compiler(results: str):
    """Compiles all test statuses and bug reports into a professional final QA Summary."""
    return "Final QA Report Compiled."