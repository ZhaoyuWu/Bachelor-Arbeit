import { createStore } from 'redux';

const initialState = {
  displayedTitle: 'N',
  pathData: [],
  penStrokes: [],
  offset: { x: 0, y: 0 },
  loading: false,
  direction: 'adding',
  index: 1,
  isPaused: false,
};

const reducer = (state = initialState, action) => {
  switch (action.type) {
    case 'SET_DISPLAYED_TITLE':
      return { ...state, displayedTitle: action.payload };
    case 'SET_PATH_DATA':
      return { ...state, pathData: action.payload };
    case 'SET_PEN_STROKES':
      return { ...state, penStrokes: action.payload };
    case 'SET_LOADING':
      return { ...state, loading: action.payload };
    case 'SET_DIRECTION':
      return { ...state, direction: action.payload };
    case 'SET_INDEX':
      return { ...state, index: action.payload };
    case 'SET_PAUSED':
      return { ...state, isPaused: action.payload };
    case 'SET_OFFSET':
      return { ...state, offset: action.payload };
    default:
      return state;
  }
};

const store = createStore(reducer);

export default store;
