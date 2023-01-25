import React, { useEffect, useState } from 'react';
import './App.css';
import '@fontsource/roboto/400.css';

import { Tooltip, LinearProgress, Button, Box, Grid, TextField, IconButton, Dialog, Snackbar, Alert } from '@mui/material'

import { ThemeProvider, createTheme, useTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';

import { post, getStatus, getPredictions } from './api';
import { NextWordButton, LabSelector } from './components';
import { RefreshOutlined, WifiTethering, WifiTetheringError } from '@mui/icons-material';

import { handleNextWord } from './util';

const darkTheme = createTheme({ palette: { mode: 'dark' } });
const lightTheme = createTheme({ palette: { mode: 'light' } });
const themes = [lightTheme, darkTheme];

const HOST = "localhost" 
// change to your local ip to access from other devices
// const HOST = "10.24.23.203"
const PORT = "8080"
const URL = `http://${HOST}:${PORT}`

const localStoredLab = localStorage.getItem("labNumber")

function App() {
  const theme = useTheme();
  const [theme_id, setThemeId] = useState(1);  // 0 = light, 1 = dark

  const [trained, setTrained] = useState(false);
  const [training, setTraining] = useState(false);

  const [labNumber, setLabNumber] = useState(localStoredLab ? parseInt(localStoredLab) : 0);
  const [predictedWords, setPredictedWords] = useState([]);
  const [inputText, setInput] = useState("");
  const [url, setUrl] = useState(URL);
  const [apiCalls, setApiCalls] = useState(0);
  const [settingsOpen, setSettingsOpen] = useState(false);

  // error handling for the server
  const [isTyping, setIsTyping] = useState(false);
  const [error, setError] = useState(null);
  
  // status messages for different server states
  const [statusMessage, setStatusMessage] = useState(null);

  // get status from the status endpoint to populate the current text and to check if the model is trained
  useEffect(() => {
    getStatus(url, setInput, setTrained, setError, labNumber);
  }, [url, labNumber]);

  useEffect(() => {
    if (!trained) { return; }
    setTimeout(() => {
      if (!isTyping) {
        getPredictions(url, inputText, labNumber, setError, setPredictedWords);
        setApiCalls(calls => calls + 1);
      }
    }, 200);
  }, [inputText, trained, isTyping, settingsOpen, url, labNumber]);

  useEffect(() => {
    if (!trained) { return; }

    let typingTimer;
    const doneTyping = () => setIsTyping(false);
    const onTyping = () => {
      clearTimeout(typingTimer);
      setIsTyping(true);
      typingTimer = setTimeout(doneTyping, 500);
    };
    const inputField = document.getElementById('inputText');
    inputField.addEventListener('keydown', onTyping);
    return () => inputField.removeEventListener('keydown', onTyping);
  }, [isTyping, trained]);

  // whether to titlecase a button and its resulting text
  const titleCased = (inputText.trim().length === 0 || inputText.slice(-1) === " ") && (
      inputText.split(" ").length <= 1 ||
      "?!.".split('').includes(inputText.trimEnd().slice(-1))
  )

  return (
    <ThemeProvider theme={themes[theme_id]}>
      <CssBaseline />
      <Grid
        container
        className="main"
        direction="column"
        justifyContent="space-between"
        style={{ height: "80vh" }}
      >
        <Box padding={1} width={1}>
          <Grid container justifyContent="space-around" alignItems="center" spacing={1}>
              {!trained || predictedWords.length === 0 ? (
                <Button disabled>No predictions received</Button>
              ):(
                <React.Fragment>
                  {[...predictedWords].map((_, i) => (
                    <Grid item key={predictedWords[i] + i}>
                      <NextWordButton
                        titleCased={titleCased}
                        word={predictedWords[i]}
                        onClick={
                          () => handleNextWord(setInput, inputText, predictedWords[i], titleCased)
                        }
                      />
                    </Grid>
                  ))}
                </React.Fragment>
              )}
          </Grid>
          <Box marginTop={2} width={1}>
            {training ? (
              <React.Fragment>
                <LinearProgress style={{ height: 50 }} />
                <p>Training...</p>
              </React.Fragment>
            ) : (
              <React.Fragment>
                {trained ? (
                  <TextField
                    id="inputText"
                    value={inputText}
                    onChange={(e) => setInput(e.target.value)}
                    variant='outlined'
                    multiline
                    fullWidth
                  />
                ) : (
                  <React.Fragment>
                    <p>{`The model for Lab ${labNumber + 1} is not trained`}</p>
                    <Button
                      onClick={async () => {
                        setTraining(true)
                        const body = JSON.stringify({"lab": labNumber})
                        const trained = await post(`${url}/train`, body)
                        setTrained(trained)
                        setTraining(false)
                      }}
                      color="inherit"
                      variant="outlined">Click to train!</Button>
                  </React.Fragment>
                )}
              </React.Fragment>
            )}
          </Box>
          <Box marginTop={2}>
            {!training && (
              <LabSelector
                labNumber={labNumber}
                onChange={async (e) => {
                  setLabNumber(e.target.value)
                  // update lab number in localstorage:
                  localStorage.setItem("labNumber", e.target.value)
                  const labChangeInfo = await post(
                    `${url}/lab`,
                    JSON.stringify({"lab": e.target.value}))
                  setStatusMessage(labChangeInfo["message"])
                }}
              />
            )}
          </Box>
        </Box>

        <Grid container justifyContent="center">
          <Button onClick={() => window.location.reload()} color="inherit">
            Reload&nbsp;<RefreshOutlined />
          </Button>
          <Button sx={{ ml: 1 }} onClick={() => setSettingsOpen(true)} color="inherit">
            Open API settings
          </Button>
          {settingsOpen && (
            <Dialog open={settingsOpen} onClose={() => setSettingsOpen(false)}>
              <Grid container width={1} padding={5} direction="column" justifyContent="center">
                <h1>API info</h1>
                <p>Endpoint URL - verify from the python server</p>
                <TextField
                  id="endpoint-field"
                  value={url}
                  variant='outlined'
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="Endpoint URL - verify from the python server"
                  fullWidth
                />
                <p>calls: {apiCalls}</p>
              </Grid>
            </Dialog>
          )}

          <Tooltip title={`Toggle ${theme_id ? 'eye killer' : 'comfy'} mode`}>
            <IconButton sx={{ ml: 1 }} onClick={() => setThemeId(theme_id === 0 ? 1 : 0)} color="inherit">
              {theme.palette.mode === 'dark' ? <Brightness7Icon /> : <Brightness4Icon />}
            </IconButton>
          </Tooltip>

          <Tooltip title={error ? "Failed to connect to server" : "Connected to server"}>
            <IconButton color="inherit">
              {error ?
                <WifiTetheringError style={{color: "red"}} />
                : <WifiTethering style={{color: "limegreen"}} />
              }
            </IconButton>
          </Tooltip>
        </Grid>
      </Grid>
    <Snackbar
      open={error !== null}
      autoHideDuration={5000}
      onClose={() => setError(null)}
    >
      <Alert severity="error">{error}</Alert>
    </Snackbar>
    <Snackbar
      open={statusMessage !== null}
      autoHideDuration={2000}
      onClose={() => setStatusMessage(null)}
    >
      <Alert severity="info">{statusMessage}</Alert>
    </Snackbar>
    </ThemeProvider>
  );
}

export default App;
