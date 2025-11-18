import type { ThemingProps } from "../../types";
import type { TimelineTheme } from "./Timeline";
export interface TimelineContextValue extends ThemingProps<TimelineTheme> {
    horizontal?: boolean;
}
export declare const TimelineContext: import("react").Context<TimelineContextValue | undefined>;
export declare function useTimelineContext(): TimelineContextValue;
