export function handleNextWord(setFunction, inputText, nextWord, titleCased) {
  const nospace_before = ".,!?;:()[]{}\"'”“’‘-–—…/\\#".split('')
  const nospace_after = "'-#([/\\"
  // if the last character typed is an apostrophe or hyphen, don't add a space
  const lastChar = inputText.slice(-1)
  inputText = inputText.trimEnd()

  let space = " "
  if (nospace_after.includes(lastChar) || nospace_before.includes(nextWord) || inputText === '') {
    space = ""
  }
  if (titleCased) {
    nextWord = nextWord.charAt(0).toUpperCase() + nextWord.slice(1)
  }
  setFunction(`${inputText}${space}${nextWord}`)
}