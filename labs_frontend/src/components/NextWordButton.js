
import { Button } from '@mui/material'

const NextWordButton = ({word, onClick, titleCased}) => (
    <Button
        onClick={onClick}
        variant="contained"
        style={{ textTransform: 'none' }}
    >
        {titleCased ? word[0].toUpperCase() + word.slice(1) : word}
    </Button>
)

export default NextWordButton
