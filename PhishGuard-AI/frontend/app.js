const API_URL = "/api/predict";

const form = document.querySelector("#scan-form");
const input = document.querySelector("#url-input");
const button = document.querySelector("#scan-button");
const buttonLabel = button.querySelector(".button-label");
const message = document.querySelector("#form-message");
const resultSection = document.querySelector("#result-section");
const resultCard = document.querySelector("#result-card");
const scannedUrl = document.querySelector("#scanned-url");
const predictionText = document.querySelector("#prediction-text");
const assessmentKicker = document.querySelector("#assessment-kicker");
const riskScore = document.querySelector("#risk-score");
const riskLevel = document.querySelector("#risk-level");
const confidence = document.querySelector("#confidence");
const confidenceLabel = document.querySelector("#confidence-label");
const detectionMethod = document.querySelector("#detection-method");
const riskMeter = document.querySelector("#risk-meter");
const meterFill = document.querySelector("#meter-fill");
const reasonsList = document.querySelector("#reasons-list");
const noReasons = document.querySelector("#no-reasons");
const resultIcon = document.querySelector("#result-icon");

function showMessage(text) {
  message.textContent = text;
}

function showResult(url, result) {
  const score = Number(result.risk_score);
  const confidenceScore = Number(result.confidence);

  if (
    !Number.isFinite(score) ||
    score < 0 ||
    score > 100 ||
    !Number.isFinite(confidenceScore) ||
    !Array.isArray(result.reasons)
  ) {
    throw new Error("The API returned an invalid prediction result.");
  }

  const isPhishing = result.prediction === "Phishing";
  resultCard.classList.toggle("is-phishing", isPhishing);
  resultIcon.textContent = isPhishing ? "!" : "✓";
  predictionText.textContent = isPhishing
    ? "PHISHING DETECTED"
    : "LEGITIMATE";
  riskScore.textContent = String(score);
  riskLevel.textContent = result.risk_level;
  confidence.textContent = String(confidenceScore);
  const method = result.detection_method || "Random Forest Machine Learning";
  detectionMethod.textContent = method;
  assessmentKicker.textContent =
    method === "Trusted Official Domain Verification"
      ? "Trusted-domain verification"
      : "Machine learning assessment";
  confidenceLabel.textContent =
    method === "Trusted Official Domain Verification"
      ? "Verification confidence"
      : "Model confidence";
  scannedUrl.textContent = url;
  riskMeter.setAttribute("aria-valuenow", String(score));
  meterFill.style.width = `${score}%`;

  reasonsList.replaceChildren();
  for (const reason of result.reasons) {
    const item = document.createElement("li");
    item.textContent = String(reason);
    reasonsList.append(item);
  }
  noReasons.hidden = result.reasons.length > 0;
  reasonsList.hidden = result.reasons.length === 0;
  resultSection.hidden = false;
  resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  showMessage("");

  const url = input.value.trim();
  if (!url) {
    showMessage("Please enter a URL.");
    input.focus();
    return;
  }

  button.disabled = true;
  buttonLabel.textContent = "Analyzing URL...";
  button.setAttribute("aria-busy", "true");

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });

    let result;
    try {
      result = await response.json();
    } catch {
      throw new Error("The server returned an unreadable response.");
    }

    if (!response.ok) {
      throw new Error(
        result.error || "Unable to analyze this URL. Please check the input.",
      );
    }
    showResult(url, result);
  } catch (error) {
    if (error instanceof TypeError) {
      showMessage(
        "Unable to connect to PhishGuard AI server. Please make sure the Flask API is running.",
      );
    } else {
      showMessage(error.message);
    }
  } finally {
    button.disabled = false;
    buttonLabel.textContent = "Scan URL";
    button.removeAttribute("aria-busy");
  }
});

document.querySelectorAll("[data-example]").forEach((exampleButton) => {
  exampleButton.addEventListener("click", () => {
    input.value = exampleButton.dataset.example;
    input.focus();
  });
});
