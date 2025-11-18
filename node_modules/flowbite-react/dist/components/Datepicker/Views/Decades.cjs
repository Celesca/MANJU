'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var tailwindMerge = require('../../../helpers/tailwind-merge.cjs');
var DatepickerContext = require('../DatepickerContext.cjs');
var helpers = require('../helpers.cjs');

function DatepickerViewsDecades() {
  const {
    theme: rootTheme,
    viewDate,
    selectedDate,
    minDate,
    maxDate,
    filterDate,
    setViewDate,
    setView
  } = DatepickerContext.useDatePickerContext();
  const theme = rootTheme.views.decades;
  const first = helpers.startOfYearPeriod(viewDate, 100);
  return /* @__PURE__ */ jsxRuntime.jsx("div", { className: theme.items.base, children: [...Array(12)].map((_year, index) => {
    const year = first - 10 + index * 10;
    const newDate = new Date(viewDate.getTime());
    newDate.setFullYear(year + viewDate.getFullYear() % 10);
    const firstDate = new Date(year, 0, 1);
    const lastDate = helpers.addYears(firstDate, 9);
    const isSelected = selectedDate && helpers.isDateInDecade(selectedDate, year);
    const isDisabled = !helpers.isDateInRange(firstDate, minDate, maxDate) && !helpers.isDateInRange(lastDate, minDate, maxDate) || filterDate && !filterDate(newDate, helpers.Views.Decades);
    return /* @__PURE__ */ jsxRuntime.jsx(
      "button",
      {
        disabled: isDisabled,
        type: "button",
        className: tailwindMerge.twMerge(
          theme.items.item.base,
          isSelected && theme.items.item.selected,
          isDisabled && theme.items.item.disabled
        ),
        onClick: () => {
          if (isDisabled) return;
          selectedDate && setViewDate(helpers.addYears(viewDate, year - selectedDate.getFullYear()));
          setView(helpers.Views.Years);
        },
        children: year
      },
      index
    );
  }) });
}
DatepickerViewsDecades.displayName = "DatepickerViewsDecades";

exports.DatepickerViewsDecades = DatepickerViewsDecades;
//# sourceMappingURL=Decades.cjs.map
