'use client';
import { createContext, useContext } from 'react';

const ButtonGroupContext = createContext(void 0);
function useButtonGroupContext() {
  return useContext(ButtonGroupContext);
}

export { ButtonGroupContext, useButtonGroupContext };
//# sourceMappingURL=ButtonGroupContext.js.map
