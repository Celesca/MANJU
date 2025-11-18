import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';

const ChevronRightIcon = forwardRef((props, ref) => /* @__PURE__ */ jsx(
  "svg",
  {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 2,
    viewBox: "0 0 24 24",
    ref,
    ...props,
    children: /* @__PURE__ */ jsx("path", { strokeLinecap: "round", strokeLinejoin: "round", d: "m9 5 7 7-7 7" })
  }
));
ChevronRightIcon.displayName = "ChevronRightIcon";

export { ChevronRightIcon };
//# sourceMappingURL=chevron-right-icon.js.map
