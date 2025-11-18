'use client';
'use strict';

var React = require('react');

const ButtonGroupContext = React.createContext(void 0);
function useButtonGroupContext() {
  return React.useContext(ButtonGroupContext);
}

exports.ButtonGroupContext = ButtonGroupContext;
exports.useButtonGroupContext = useButtonGroupContext;
//# sourceMappingURL=ButtonGroupContext.cjs.map
