export async function post(url, body) {
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body,
    }).then((res) => res.json())
      .catch((err) => console.error(err));
}

export async function getStatus(url, setInput, setTrained, setError, labNumber) {
  try {
    let statusUrl = `${url}/status`;
    const status = await fetch(statusUrl);
    const statusJson = await status.json();
    setInput(statusJson.text);
    setTrained(statusJson.trained[labNumber]);
  }
  catch (e) {
    setError(`Could not connect to server. Verify the URL (${url})`)
  }
}

export async function getPredictions(url, inputText, labNumber, setError, setPredictedWords) {
  try {
    let predictionUrl = `${url}/predictions`;
    let body = JSON.stringify({"text": inputText, "lab": labNumber})
    const words = await post(predictionUrl, body);
    if (!Array.isArray(words)) {
      setError(`No predictions received. Verify the URL (${url})`)
      setPredictedWords([]);
    }
    else {
      setError(null);
      setPredictedWords(words);
    }
  }
  catch (e) {
    setError(`Could not connect to server. Verify the URL (${url})`)
  }
}