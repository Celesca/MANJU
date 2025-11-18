import type { ThemingProps } from "../../types";
import type { TimelineItemTheme } from "./TimelineItem";
export type TimelineItemContextValue = ThemingProps<TimelineItemTheme>;
export declare const TimelineItemContext: import("react").Context<TimelineItemContextValue | undefined>;
export declare function useTimelineItemContext(): TimelineItemContextValue;
