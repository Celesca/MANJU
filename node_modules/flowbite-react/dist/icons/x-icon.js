import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';

const XIcon = forwardRef((props, ref) => /* @__PURE__ */ jsx(
  "svg",
  {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "currentColor",
    stroke: "currentColor",
    strokeWidth: 0,
    viewBox: "0 0 20 20",
    ref,
    ...props,
    children: /* @__PURE__ */ jsx(
      "path",
      {
        fillRule: "evenodd",
        stroke: "none",
        d: "M4.293 4.293a1 1 0 0 1 1.414 0L10 8.586l4.293-4.293a1 1 0 1 1 1.414 1.414L11.414 10l4.293 4.293a1 1 0 0 1-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 0 1-1.414-1.414L8.586 10 4.293 5.707a1 1 0 0 1 0-1.414z",
        clipRule: "evenodd"
      }
    )
  }
));
XIcon.displayName = "XIcon";

export { XIcon };
//# sourceMappingURL=x-icon.js.map
