import { FormControl, Select, MenuItem } from '@mui/material'
import { NUM_LABS } from '../constants'

const LabSelector = ({labNumber, onChange}) => (
  <FormControl variant="outlined">
    <Select
      id="lab-selector"
      value={labNumber}
      label="Lab number"
      onChange={onChange}
    >
      {Array.from(Array(NUM_LABS).keys()).map((e) => (
        <MenuItem value={e} key={e}>Lab {e + 1}</MenuItem>
      ))}
    </Select>
  </FormControl>
)

export default LabSelector
