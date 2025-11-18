'use client';
import { forwardRef, createElement } from 'react';

const ButtonBase = forwardRef(
  ({ children, as: Component, href, type = "button", ...props }, ref) => {
    const BaseComponent = Component || (href ? "a" : "button");
    return createElement(BaseComponent, { ref, href, type, ...props }, children);
  }
);
ButtonBase.displayName = "ButtonBase";

export { ButtonBase };
//# sourceMappingURL=ButtonBase.js.map
