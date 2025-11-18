'use client';
import { jsx } from 'react/jsx-runtime';
import { twMerge } from '../../../helpers/tailwind-merge.js';
import { useDatePickerContext } from '../DatepickerContext.js';
import { startOfYearPeriod, isDateEqual, isDateInRange, Views } from '../helpers.js';

function DatepickerViewsYears() {
  const {
    theme: rootTheme,
    selectedDate,
    minDate,
    maxDate,
    filterDate,
    viewDate,
    setViewDate,
    setView
  } = useDatePickerContext();
  const theme = rootTheme.views.years;
  return /* @__PURE__ */ jsx("div", { className: theme.items.base, children: [...Array(12)].map((_year, index) => {
    const first = startOfYearPeriod(viewDate, 10);
    const year = first + index;
    const newDate = new Date(viewDate.getTime());
    newDate.setFullYear(year);
    const isSelected = selectedDate && isDateEqual(selectedDate, newDate);
    const isDisabled = !isDateInRange(newDate, minDate, maxDate) || filterDate && !filterDate(newDate, Views.Years);
    return /* @__PURE__ */ jsx(
      "button",
      {
        disabled: isDisabled,
        type: "button",
        className: twMerge(
          theme.items.item.base,
          isSelected && theme.items.item.selected,
          isDisabled && theme.items.item.disabled
        ),
        onClick: () => {
          if (isDisabled) return;
          setViewDate(newDate);
          setView(Views.Months);
        },
        children: year
      },
      index
    );
  }) });
}
DatepickerViewsYears.displayName = "DatepickerViewsYears";

export { DatepickerViewsYears };
//# sourceMappingURL=Years.js.map
