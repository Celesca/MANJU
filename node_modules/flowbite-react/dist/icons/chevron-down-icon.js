import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';

const ChevronDownIcon = forwardRef((props, ref) => /* @__PURE__ */ jsx(
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
        d: "M5.293 7.293a1 1 0 0 1 1.414 0L10 10.586l3.293-3.293a1 1 0 1 1 1.414 1.414l-4 4a1 1 0 0 1-1.414 0l-4-4a1 1 0 0 1 0-1.414z",
        clipRule: "evenodd"
      }
    )
  }
));
ChevronDownIcon.displayName = "ChevronDownIcon";

export { ChevronDownIcon };
//# sourceMappingURL=chevron-down-icon.js.map
