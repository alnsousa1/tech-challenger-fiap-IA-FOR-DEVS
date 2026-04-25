const featureGroupsContainer = document.getElementById("feature-groups");
const modelComparisonContainer = document.getElementById("model-comparison");
const metricGrid = document.getElementById("metric-grid");
const resultBody = document.getElementById("result-body");
const riskPill = document.getElementById("risk-pill");
const championModel = document.getElementById("champion-model");
const championRecall = document.getElementById("champion-recall");
const championF1 = document.getElementById("champion-f1");
const metricRationale = document.getElementById("metric-rationale");
const predictionResultsPanel = document.getElementById("prediction-results");
const scrollToResultButton = document.getElementById("scroll-to-result");

let modelInfo = null;

const locale = "pt-BR";

const MODEL_LABELS = {
  logistic_regression: "Regressão Logística",
  decision_tree: "Árvore de Decisão",
};

const GROUP_LABELS = {
  mean_measurements: "Medidas médias",
  measurement_error: "Erros de medição",
  worst_case_measurements: "Piores medidas",
};

const FEATURE_LABELS = {
  mean_radius: "Raio médio",
  mean_texture: "Textura média",
  mean_perimeter: "Perímetro médio",
  mean_area: "Área média",
  mean_smoothness: "Suavidade média",
  mean_compactness: "Compacidade média",
  mean_concavity: "Concavidade média",
  mean_concave_points: "Pontos côncavos médios",
  mean_symmetry: "Simetria média",
  mean_fractal_dimension: "Dimensão fractal média",
  radius_error: "Erro do raio",
  texture_error: "Erro da textura",
  perimeter_error: "Erro do perímetro",
  area_error: "Erro da área",
  smoothness_error: "Erro da suavidade",
  compactness_error: "Erro da compacidade",
  concavity_error: "Erro da concavidade",
  concave_points_error: "Erro dos pontos côncavos",
  symmetry_error: "Erro da simetria",
  fractal_dimension_error: "Erro da dimensão fractal",
  worst_radius: "Pior raio",
  worst_texture: "Pior textura",
  worst_perimeter: "Pior perímetro",
  worst_area: "Pior área",
  worst_smoothness: "Pior suavidade",
  worst_compactness: "Pior compacidade",
  worst_concavity: "Pior concavidade",
  worst_concave_points: "Piores pontos côncavos",
  worst_symmetry: "Pior simetria",
  worst_fractal_dimension: "Pior dimensão fractal",
  area_perimeter_ratio: "Razão área/perímetro",
  compactness_to_concavity_ratio: "Razão compacidade/concavidade",
  radius_error_pct: "Erro percentual do raio",
  worst_radius_delta: "Diferença do pior raio",
  radius_band_small: "Faixa de raio: pequeno",
  radius_band_medium: "Faixa de raio: médio",
  radius_band_large: "Faixa de raio: grande",
  texture_band_low: "Faixa de textura: baixa",
  texture_band_moderate: "Faixa de textura: moderada",
  texture_band_high: "Faixa de textura: alta",
  concavity_band_mild: "Faixa de concavidade: leve",
  concavity_band_elevated: "Faixa de concavidade: elevada",
  concavity_band_severe: "Faixa de concavidade: severa",
};

const CLASS_LABELS = {
  benign: "Benigno",
  malignant: "Maligno",
};

const RISK_LABELS = {
  low: "Baixo",
  moderate: "Moderado",
  high: "Alto",
};

const CONTRIBUTOR_DIRECTION_LABELS = {
  increases_risk: "aumenta o risco",
  decreases_risk: "reduz o risco",
};

const UI_TEXT = {
  optional: "Opcional",
  champion: "Campeão",
  candidate: "Candidato",
  accuracy: "Acurácia",
  recall: "Recall",
  f1Score: "F1-score",
  predictionTitlePrefix: "Predição",
  malignantProbability: "Probabilidade de malignidade",
  benignProbability: "Probabilidade de benignidade",
  threshold: "Limiar",
  missingValuesImputed: "Valores ausentes imputados",
  metricRationale:
    "O recall é priorizado porque um falso negativo em um caso maligno pode atrasar o diagnóstico. O F1-score complementa essa visão ao equilibrar sensibilidade clínica e custo operacional dos falsos positivos.",
  defaultDisclaimer:
    "Esta estimativa apoia a triagem e a interpretação do risco. Ela não substitui avaliação clínica nem confirmação diagnóstica.",
  genericPredictionError: "Não foi possível concluir a predição.",
  genericLoadError: "Não foi possível carregar as informações do modelo.",
  emptyPayloadError: "Informe ao menos uma medição para realizar a predição.",
  predictionTimestamp: "Predição realizada em",
  attemptTimestamp: "Tentativa registrada em",
  predictionCompleted: "Predição concluída",
  predictionCompletedMessage: "A predição foi realizada com sucesso às",
  predictionErrorTitle: "Erro na predição",
  predictionErrorMessage: "Ocorreu um erro ao processar a predição. Revise os dados informados e tente novamente.",
  scrollToResult: "Voltar ao resultado da predição",
  resultReady: "Resultado pronto",
  resultError: "Falha",
};

function humanize(text) {
  return text.replaceAll("_", " ").replace(/\b\w/g, (character) => character.toUpperCase());
}

function formatDecimal(value, fractionDigits = 3) {
  return Number(value).toLocaleString(locale, {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  });
}

function formatPercentage(value) {
  return Number(value).toLocaleString(locale, {
    style: "percent",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function formatTimestamp(date = new Date()) {
  return new Intl.DateTimeFormat(locale, {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}

function translateModelName(name) {
  return MODEL_LABELS[name] ?? humanize(name);
}

function translateGroupName(name) {
  return GROUP_LABELS[name] ?? humanize(name);
}

function translateFeatureName(name) {
  return FEATURE_LABELS[name] ?? humanize(name);
}

function translateClassName(name) {
  return CLASS_LABELS[name] ?? humanize(name);
}

function translateRiskLevel(level) {
  return RISK_LABELS[level] ?? humanize(level);
}

function translateDirection(direction) {
  return CONTRIBUTOR_DIRECTION_LABELS[direction] ?? humanize(direction);
}

function translateErrorMessage(message) {
  if (!message) {
    return UI_TEXT.genericPredictionError;
  }

  if (message.includes("At least one measurement must be provided")) {
    return UI_TEXT.emptyPayloadError;
  }

  return message;
}

function showAlert(title, message, type = "success") {
  if (typeof Swal === "undefined") {
    window.alert(`${title}\n\n${message}`);
    return;
  }

  Swal.fire({
    toast: true,
    position: "top-end",
    icon: type,
    title,
    text: message,
    showConfirmButton: false,
    timer: 4200,
    timerProgressBar: true,
    customClass: {
      popup: "swal-health-popup",
      title: "swal-health-title",
      htmlContainer: "swal-health-text",
      icon: "swal-health-icon",
    },
  });
}

function revealScrollButton() {
  scrollToResultButton.hidden = false;
}

function modelIcon(name) {
  if (name === "logistic_regression") {
    return "bi bi-bezier2";
  }

  if (name === "decision_tree") {
    return "bi bi-diagram-3-fill";
  }

  return "bi bi-cpu-fill";
}

function createField(name) {
  const wrapper = document.createElement("div");
  wrapper.className = "field";

  const label = document.createElement("label");
  label.htmlFor = name;
  label.textContent = translateFeatureName(name);

  const input = document.createElement("input");
  input.type = "number";
  input.step = "any";
  input.id = name;
  input.name = name;
  input.placeholder = UI_TEXT.optional;

  wrapper.append(label, input);
  return wrapper;
}

function renderFeatureGroups(groups) {
  featureGroupsContainer.innerHTML = "";
  Object.entries(groups).forEach(([groupName, features]) => {
    const section = document.createElement("section");
    section.className = "feature-group";

    const heading = document.createElement("h3");
    heading.textContent = translateGroupName(groupName);

    const grid = document.createElement("div");
    grid.className = "feature-grid";
    features.forEach((feature) => grid.appendChild(createField(feature)));

    section.append(heading, grid);
    featureGroupsContainer.appendChild(section);
  });
}

function renderModelComparison(models, championName) {
  modelComparisonContainer.innerHTML = "";
  Object.entries(models).forEach(([modelName, metrics]) => {
    const card = document.createElement("article");
    card.className = "comparison-card";
    card.innerHTML = `
      <div class="comparison-head">
        <div>
          <span class="stat-label">${modelName === championName ? UI_TEXT.champion : UI_TEXT.candidate}</span>
          <h3>${translateModelName(modelName)}</h3>
        </div>
        <span class="comparison-icon">
          <i class="${modelIcon(modelName)}"></i>
        </span>
      </div>
      <p>${UI_TEXT.accuracy} <strong>${formatDecimal(metrics.accuracy)}</strong></p>
      <p>${UI_TEXT.recall} <strong>${formatDecimal(metrics.recall)}</strong></p>
      <p>${UI_TEXT.f1Score} <strong>${formatDecimal(metrics.f1_score)}</strong></p>
    `;
    modelComparisonContainer.appendChild(card);
  });
}

function fillForm(sample) {
  Object.entries(sample).forEach(([name, value]) => {
    const field = document.getElementById(name);
    if (field) {
      field.value = value ?? "";
    }
  });
}

async function loadModelInfo() {
  const response = await fetch("/model-info");
  modelInfo = await response.json();

  renderFeatureGroups(modelInfo.feature_groups);
  renderModelComparison(modelInfo.models, modelInfo.champion_model.name);

  championModel.textContent = translateModelName(modelInfo.champion_model.name);
  championRecall.textContent = formatDecimal(modelInfo.champion_model.metrics.recall);
  championF1.textContent = formatDecimal(modelInfo.champion_model.metrics.f1_score);
  metricRationale.textContent = UI_TEXT.metricRationale;

  document.getElementById("load-benign").addEventListener("click", () => {
    fillForm(modelInfo.sample_payloads.benign_example);
  });

  document.getElementById("load-malignant").addEventListener("click", () => {
    fillForm(modelInfo.sample_payloads.malignant_example);
  });
}

function serializeForm(form) {
  const payload = {};
  new FormData(form).forEach((value, key) => {
    payload[key] = value === "" ? null : Number(value);
  });
  return payload;
}

function renderPrediction(result, timestampLabel) {
  riskPill.textContent = translateRiskLevel(result.risk_level).toUpperCase();
  riskPill.className = `pill ${result.risk_level}`;

  resultBody.innerHTML = `
    <div class="result-title-row">
      <div>
        <h3>${UI_TEXT.predictionTitlePrefix}: ${translateClassName(result.predicted_class)}</h3>
        <p class="result-meta">${UI_TEXT.predictionTimestamp} ${timestampLabel}</p>
      </div>
      <span class="result-status success">${UI_TEXT.resultReady}</span>
    </div>
    <p>${UI_TEXT.defaultDisclaimer}</p>
    <ul class="contributors">
      ${result.top_contributors
        .map(
          (item) =>
            `<li><strong>${translateFeatureName(item.feature.toLowerCase().replaceAll(" ", "_"))}</strong> (${translateDirection(item.direction)}) impacto ${formatDecimal(item.impact, 4)}</li>`
        )
        .join("")}
    </ul>
  `;

  metricGrid.innerHTML = `
    <article class="metric-card">
      <span class="stat-label">${UI_TEXT.malignantProbability}</span>
      <strong>${formatPercentage(result.malignant_probability)}</strong>
    </article>
    <article class="metric-card">
      <span class="stat-label">${UI_TEXT.benignProbability}</span>
      <strong>${formatPercentage(result.benign_probability)}</strong>
    </article>
    <article class="metric-card">
      <span class="stat-label">${UI_TEXT.threshold}</span>
      <strong>${formatDecimal(result.threshold, 2)}</strong>
    </article>
    <article class="metric-card">
      <span class="stat-label">${UI_TEXT.missingValuesImputed}</span>
      <strong>${result.missing_features_imputed.length}</strong>
    </article>
  `;
}

function renderPredictionError(message, timestampLabel) {
  riskPill.textContent = "ERRO";
  riskPill.className = "pill error";
  resultBody.innerHTML = `
    <div class="result-title-row">
      <div>
        <h3>${UI_TEXT.predictionErrorTitle}</h3>
        <p class="result-meta">${UI_TEXT.attemptTimestamp} ${timestampLabel}</p>
      </div>
      <span class="result-status error">${UI_TEXT.resultError}</span>
    </div>
    <p>${message}</p>
  `;
  metricGrid.innerHTML = "";
}

document.getElementById("prediction-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = serializeForm(event.currentTarget);
  const timestampLabel = formatTimestamp();

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const error = await response.json();
      const translatedError = translateErrorMessage(error.detail?.[0]?.msg) ?? UI_TEXT.genericPredictionError;
      renderPredictionError(translatedError, timestampLabel);
      showAlert(UI_TEXT.predictionErrorTitle, translatedError, "error");
      revealScrollButton();
      return;
    }

    const result = await response.json();
    renderPrediction(result, timestampLabel);
    showAlert(UI_TEXT.predictionCompleted, `${UI_TEXT.predictionCompletedMessage} ${timestampLabel}.`, "success");
    revealScrollButton();
  } catch (error) {
    console.error(error);
    renderPredictionError(UI_TEXT.predictionErrorMessage, timestampLabel);
    showAlert(UI_TEXT.predictionErrorTitle, UI_TEXT.predictionErrorMessage, "error");
    revealScrollButton();
  }
});

scrollToResultButton.addEventListener("click", () => {
  predictionResultsPanel.scrollIntoView({
    behavior: "smooth",
    block: "start",
  });
});

loadModelInfo().catch((error) => {
  console.error(error);
  resultBody.innerHTML = `<p>${UI_TEXT.genericLoadError}</p>`;
  showAlert(UI_TEXT.predictionErrorTitle, UI_TEXT.genericLoadError, "error");
});

window.addEventListener("load", () => {
  if (typeof AOS !== "undefined") {
    AOS.init({
      duration: 720,
      once: true,
      easing: "ease-out-cubic",
      offset: 40,
    });
  }
});
