'use client';
'use strict';

var React = require('react');

const ButtonBase = React.forwardRef(
  ({ children, as: Component, href, type = "button", ...props }, ref) => {
    const BaseComponent = Component || (href ? "a" : "button");
    return React.createElement(BaseComponent, { ref, href, type, ...props }, children);
  }
);
ButtonBase.displayName = "ButtonBase";

exports.ButtonBase = ButtonBase;
//# sourceMappingURL=ButtonBase.cjs.map
