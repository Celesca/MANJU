import type { ThemingProps } from "../../types";
import type { TimelineContentTheme } from "./TimelineContent";
export type TimelineContentContextValue = ThemingProps<TimelineContentTheme>;
export declare const TimelineContentContext: import("react").Context<TimelineContentContextValue | undefined>;
export declare function useTimelineContentContext(): TimelineContentContextValue;
